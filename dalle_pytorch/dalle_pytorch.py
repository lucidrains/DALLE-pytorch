from math import log2, sqrt
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange
from axial_positional_embedding import AxialPositionalEmbedding
from vector_quantize_pytorch import VectorQuantize
from dalle_pytorch.transformer import Transformer

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def always(val):
    def inner(*args, **kwargs):
        return val
    return inner

def is_empty(t):
    return t.nelement() == 0

def masked_mean(t, mask, dim = 1):
    t = t.masked_fill(~mask[:, :, None], 0.)
    return t.sum(dim = 1) / mask.sum(dim = 1)[..., None]

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# sampling helpers

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# discrete vae class

class ResBlock(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x

class DiscreteVAE(nn.Module):
    def __init__(
        self,
        image_size = 256,
        num_tokens = 512,
        codebook_dim = 512,
        num_layers = 3,
        num_resnet_blocks = 0,
        hidden_dim = 64,
        channels = 3,
        smooth_l1_loss = False,
        temperature = 0.9,
        straight_through = False,
        kl_div_loss_weight = 0.
    ):
        super().__init__()
        assert log2(image_size).is_integer(), 'image size must be a power of 2'
        assert num_layers >= 1, 'number of layers must be greater than or equal to 1'
        has_resblocks = num_resnet_blocks > 0

        self.image_size = image_size
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.temperature = temperature
        self.straight_through = straight_through
        self.codebook = nn.Embedding(num_tokens, codebook_dim)

        hdim = hidden_dim

        enc_chans = [hidden_dim] * num_layers
        dec_chans = list(reversed(enc_chans))

        enc_chans = [channels, *enc_chans]

        dec_init_chan = codebook_dim if not has_resblocks else dec_chans[0]
        dec_chans = [dec_init_chan, *dec_chans]

        enc_chans_io, dec_chans_io = map(lambda t: list(zip(t[:-1], t[1:])), (enc_chans, dec_chans))

        enc_layers = []
        dec_layers = []

        for (enc_in, enc_out), (dec_in, dec_out) in zip(enc_chans_io, dec_chans_io):
            enc_layers.append(nn.Sequential(nn.Conv2d(enc_in, enc_out, 4, stride = 2, padding = 1), nn.ReLU()))
            dec_layers.append(nn.Sequential(nn.ConvTranspose2d(dec_in, dec_out, 4, stride = 2, padding = 1), nn.ReLU()))

        for _ in range(num_resnet_blocks):
            dec_layers.insert(0, ResBlock(dec_chans[1]))
            enc_layers.append(ResBlock(enc_chans[-1]))

        if num_resnet_blocks > 0:
            dec_layers.insert(0, nn.Conv2d(codebook_dim, dec_chans[1], 1))

        enc_layers.append(nn.Conv2d(enc_chans[-1], num_tokens, 1))
        dec_layers.append(nn.Conv2d(dec_chans[-1], channels, 1))

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
        self.kl_div_loss_weight = kl_div_loss_weight

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, images):
        logits = self.forward(images, return_logits = True)
        codebook_indices = logits.argmax(dim = 1).flatten(1)
        return codebook_indices

    def decode(
        self,
        img_seq
    ):
        image_embeds = self.codebook(img_seq)
        b, n, d = image_embeds.shape
        h = w = int(sqrt(n))

        image_embeds = rearrange(image_embeds, 'b (h w) d -> b d h w', h = h, w = w)
        images = self.decoder(image_embeds)
        return images

    def forward(
        self,
        img,
        return_loss = False,
        return_recons = False,
        return_logits = False,
        temp = None
    ):
        device, num_tokens, kl_div_loss_weight = img.device, self.num_tokens, self.kl_div_loss_weight

        logits = self.encoder(img)

        if return_logits:
            return logits # return logits for getting hard image indices for DALL-E training

        temp = default(temp, self.temperature)
        soft_one_hot = F.gumbel_softmax(logits, tau = temp, dim = 1, hard = self.straight_through)
        sampled = einsum('b n h w, n d -> b d h w', soft_one_hot, self.codebook.weight)
        out = self.decoder(sampled)

        if not return_loss:
            return out

        # reconstruction loss

        recon_loss = self.loss_fn(img, out)

        # kl divergence

        logits = rearrange(logits, 'b n h w -> b (h w) n')
        qy = F.softmax(logits, dim = -1)

        log_qy = torch.log(qy + 1e-10)
        log_uniform = torch.log(torch.tensor([1. / num_tokens], device = device))
        kl_div = F.kl_div(log_uniform, log_qy, None, None, 'batchmean', log_target = True)

        loss = recon_loss + (kl_div * kl_div_loss_weight)

        if not return_recons:
            return loss

        return loss, out

class VQVAE(nn.Module):
    def __init__(
        self,
        image_size = 256,
        num_tokens = 512,
        codebook_dim = 512,
        num_layers = 3,
        num_resnet_blocks = 0,
        hidden_dim = 64,
        channels = 3,
        smooth_l1_loss = False,
        vq_decay = 0.8,
        commitment_weight = 1.
    ):
        super().__init__()
        assert log2(image_size).is_integer(), 'image size must be a power of 2'
        assert num_layers >= 1, 'number of layers must be greater than or equal to 1'
        has_resblocks = num_resnet_blocks > 0

        self.image_size = image_size
        self.num_tokens = num_tokens
        self.num_layers = num_layers

        self.vq = VectorQuantize(
            dim = codebook_dim,
            n_embed = num_tokens,
            decay = vq_decay,
            commitment = commitment_weight
        )

        hdim = hidden_dim

        enc_chans = [hidden_dim] * num_layers
        dec_chans = list(reversed(enc_chans))

        enc_chans = [channels, *enc_chans]

        dec_init_chan = codebook_dim if not has_resblocks else dec_chans[0]
        dec_chans = [dec_init_chan, *dec_chans]

        enc_chans_io, dec_chans_io = map(lambda t: list(zip(t[:-1], t[1:])), (enc_chans, dec_chans))

        enc_layers = []
        dec_layers = []

        for (enc_in, enc_out), (dec_in, dec_out) in zip(enc_chans_io, dec_chans_io):
            enc_layers.append(nn.Sequential(nn.Conv2d(enc_in, enc_out, 4, stride = 2, padding = 1), nn.ReLU()))
            dec_layers.append(nn.Sequential(nn.ConvTranspose2d(dec_in, dec_out, 4, stride = 2, padding = 1), nn.ReLU()))

        for _ in range(num_resnet_blocks):
            dec_layers.insert(0, ResBlock(dec_chans[1]))
            enc_layers.append(ResBlock(enc_chans[-1]))

        if num_resnet_blocks > 0:
            dec_layers.insert(0, nn.Conv2d(codebook_dim, dec_chans[1], 1))

        enc_layers.append(nn.Conv2d(enc_chans[-1], codebook_dim, 1))
        dec_layers.append(nn.Conv2d(dec_chans[-1], channels, 1))

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, images):
        encoded = self.forward(images, return_encoded = True)
        encoded = rearrange(encoded, 'b c h w -> b (h w) c')
        _, indices, _ = self.vq(encoded)
        return indices

    def decode(
        self,
        img_seq
    ):
        codebook = rearrange(self.vq.embed, 'd n -> n d')
        image_embeds = codebook[img_seq]
        b, n, d = image_embeds.shape
        h = w = int(sqrt(n))

        image_embeds = rearrange(image_embeds, 'b (h w) d -> b d h w', h = h, w = w)
        images = self.decoder(image_embeds)
        return images

    def forward(
        self,
        img,
        return_loss = False,
        return_recons = False,
        return_encoded = False
    ):
        shape, device = img.shape, img.device

        encoded = self.encoder(img)

        if return_encoded:
            return encoded

        h, w = encoded.shape[-2:]

        encoded = rearrange(encoded, 'b c h w -> b (h w) c')
        quantized, _, commit_loss = self.vq(encoded)
        quantized = rearrange(quantized, 'b (h w) c -> b c h w', h = h, w = w)
        out = self.decoder(quantized)

        if not return_loss:
            return out

        # reconstruction loss and VQ commitment loss

        recon_loss = self.loss_fn(img, out)

        loss = recon_loss + commit_loss

        if not return_recons:
            return loss

        return loss, out

# main classes

class CLIP(nn.Module):
    def __init__(
        self,
        *,
        dim_text = 512,
        dim_image = 512,
        dim_latent = 512,
        num_text_tokens = 10000,
        text_enc_depth = 6,
        text_seq_len = 256,
        text_heads = 8,
        num_visual_tokens = 512,
        visual_enc_depth = 6,
        visual_heads = 8,
        visual_image_size = 256,
        visual_patch_size = 32,
        channels = 3
    ):
        super().__init__()
        self.text_emb = nn.Embedding(num_text_tokens, dim_text)
        self.text_pos_emb = nn.Embedding(text_seq_len, dim_text)
        self.text_transformer = Transformer(causal = False, seq_len = text_seq_len, dim = dim_text, depth = text_enc_depth, heads = text_heads)
        self.to_text_latent = nn.Linear(dim_text, dim_latent, bias = False)

        assert visual_image_size % visual_patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (visual_image_size // visual_patch_size) ** 2
        patch_dim = channels * visual_patch_size ** 2

        self.visual_patch_size = visual_patch_size
        self.to_visual_embedding = nn.Linear(patch_dim, dim_image)
        self.visual_pos_emb = nn.Embedding(num_patches, dim_image)
        self.visual_transformer = Transformer(causal = False, seq_len = num_patches, dim = dim_image, depth = visual_enc_depth, heads = visual_heads)
        self.to_visual_latent = nn.Linear(dim_image, dim_latent, bias = False)

        self.temperature = nn.Parameter(torch.tensor(1.))

    def forward(
        self,
        text,
        image,
        text_mask = None,
        return_loss = False
    ):
        b, device, p = text.shape[0], text.device, self.visual_patch_size

        text_emb = self.text_emb(text)
        text_emb += self.text_pos_emb(torch.arange(text.shape[1], device = device))

        image_patches = rearrange(image, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        image_emb = self.to_visual_embedding(image_patches)
        image_emb += self.visual_pos_emb(torch.arange(image_emb.shape[1], device = device))

        enc_text = self.text_transformer(text_emb, mask = text_mask)
        enc_image = self.visual_transformer(image_emb)

        if exists(text_mask):
            text_latents = masked_mean(enc_text, text_mask, dim = 1)
        else:
            text_latents = enc_text.mean(dim = 1)

        image_latents = enc_image.mean(dim = 1)

        text_latents = self.to_text_latent(text_latents)
        image_latents = self.to_visual_latent(image_latents)

        text_latents, image_latents = map(lambda t: F.normalize(t, p = 2, dim = -1), (text_latents, image_latents))

        temp = self.temperature.exp()

        if not return_loss:
            sim = einsum('n d, n d -> n', text_latents, image_latents) * temp
            return sim

        sim = einsum('i d, j d -> i j', text_latents, image_latents) * temp
        labels = torch.arange(b, device = device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
        return loss

# main DALL-E class

class DALLE(nn.Module):
    def __init__(
        self,
        *,
        dim,
        vae,
        num_text_tokens = 10000,
        text_seq_len = 256,
        depth,
        heads = 8,
        dim_head = 64,
        reversible = False,
        attn_dropout = 0.,
        ff_dropout = 0,
        sparse_attn = False,
        ignore_index = -100,
        attn_types = None
    ):
        super().__init__()
        assert isinstance(vae, (DiscreteVAE, VQVAE)), 'vae must be an instance of DiscreteVAE or VQVAE'

        image_size = vae.image_size
        num_image_tokens = vae.num_tokens
        image_fmap_size = (vae.image_size // (2 ** vae.num_layers))
        image_seq_len = image_fmap_size ** 2

        self.text_emb = nn.Embedding(num_text_tokens, dim)
        self.image_emb = nn.Embedding(num_image_tokens, dim)

        self.text_pos_emb = nn.Embedding(text_seq_len + 1, dim) # +1 for <bos>
        self.image_pos_emb = AxialPositionalEmbedding(dim, axial_shape = (image_size, image_size))

        self.num_text_tokens = num_text_tokens # for offsetting logits index and calculating cross entropy loss
        self.num_image_tokens = num_image_tokens

        self.text_seq_len = text_seq_len
        self.image_seq_len = image_seq_len

        seq_len = text_seq_len + image_seq_len
        total_tokens = num_text_tokens + num_image_tokens
        self.total_tokens = total_tokens
        self.total_seq_len = seq_len

        self.vae = vae

        self.transformer = Transformer(
            dim = dim,
            causal = True,
            seq_len = seq_len,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            reversible = reversible,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            attn_types = attn_types,
            image_fmap_size = image_fmap_size,
            sparse_attn = sparse_attn
        )

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.total_tokens),
        )

        seq_range = torch.arange(seq_len)
        logits_range = torch.arange(total_tokens)

        seq_range = rearrange(seq_range, 'n -> () n ()')
        logits_range = rearrange(logits_range, 'd -> () () d')

        logits_mask = (
            ((seq_range >= text_seq_len) & (logits_range < num_text_tokens)) |
            ((seq_range < text_seq_len) & (logits_range >= num_text_tokens))
        )

        self.register_buffer('logits_mask', logits_mask)

        self.ignore_index = ignore_index

    @torch.no_grad()
    @eval_decorator
    def generate_images(
        self,
        text,
        *,
        clip = None,
        mask = None,
        filter_thres = 0.5,
        temperature = 1.
    ):
        vae, text_seq_len, image_seq_len, num_text_tokens = self.vae, self.text_seq_len, self.image_seq_len, self.num_text_tokens
        total_len = text_seq_len + image_seq_len

        out = text

        for cur_len in range(text.shape[1], total_len):
            is_image = cur_len >= text_seq_len

            text, image = out[:, :text_seq_len], out[:, text_seq_len:]

            logits = self(text, image, mask = mask)[:, -1, :]

            filtered_logits = top_k(logits, thres = filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim = -1)
            sample = torch.multinomial(probs, 1)

            sample -= (num_text_tokens if is_image else 0) # offset sampled token if it is an image token, since logit space is composed of text and then image tokens
            out = torch.cat((out, sample), dim=-1)

            if out.shape[1] <= text_seq_len:
                mask = F.pad(mask, (0, 1), value = True)

        text_seq = out[:, :text_seq_len]

        img_seq = out[:, -image_seq_len:]
        images = vae.decode(img_seq)

        if exists(clip):
            scores = clip(text_seq, images, return_loss = False)
            return images, scores

        return images

    def forward(
        self,
        text,
        image = None,
        mask = None,
        return_loss = False
    ):
        assert text.shape[-1] == self.text_seq_len, f'the length {text.shape[-1]} of the text tokens you passed in does not have the correct length ({self.text_seq_len})'
        device, ignore_index, total_seq_len = text.device, self.ignore_index, self.total_seq_len

        text = F.pad(text, (1, 0), value = 0) # use padding as <bos>

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)

        tokens = self.text_emb(text)
        tokens += self.text_pos_emb(torch.arange(text.shape[1], device = device))

        seq_len = tokens.shape[1]

        if exists(image) and not is_empty(image):
            is_raw_image = len(image.shape) == 4
            if is_raw_image:
                image = self.vae.get_codebook_indices(image)

            image_len = image.shape[1]
            image_emb = self.image_emb(image)

            image_emb += self.image_pos_emb(image_emb)

            tokens = torch.cat((tokens, image_emb), dim = 1)

            seq_len += image_len
            if exists(mask):
                mask = F.pad(mask, (0, image_emb.shape[1]), value = True)

        # when training, if the length exceeds the total text + image length
        # remove the last token, since it needs not to be trained
        if tokens.shape[1] > total_seq_len:
            seq_len -= 1
            tokens = tokens[:, :-1]

            if exists(mask):
                mask = mask[:, :-1]

        out = self.transformer(tokens, mask = mask)
        logits = self.to_logits(out)

        # mask logits to make sure text predicts text (except last token), and image predicts image
        logits_mask = self.logits_mask[:, :seq_len]
        max_neg_value = -torch.finfo(logits.dtype).max
        logits.masked_fill_(logits_mask, max_neg_value)

        if not return_loss:
            return logits

        assert exists(image), 'when training, image must be supplied'
        offsetted_image = image + self.num_text_tokens
        labels = torch.cat((text[:, 1:], offsetted_image), dim = 1)

        loss = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), labels)
        return loss
