import torch
from dalle_pytorch import OpenAIDiscreteVAE, DALLE
from matplotlib import image
from matplotlib import pyplot
import np
import pickle
import itertools



class Model():

    def __init__(self):
        self.file = "current_model.sav"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self, fileName):
        
        data = image.imread(fileName)
        # summarize shape of the pixel array
        print(data.dtype)
        print(data.shape)

            
        vae = OpenAIDiscreteVAE()       # loads pretrained OpenAI VAE

        self.dalle = DALLE(
            dim = data.shape[0],
            vae = vae,                  # automatically infer (1) image sequence length and (2) number of image tokens
            num_text_tokens = 10000,    # vocab size for text
            text_seq_len = 256,         # text sequence length
            depth = 1,                  # should aim to be 64
            heads = 16,                 # attention heads
            dim_head = 64,              # attention head dimension
            attn_dropout = 0.1,         # attention dropout
            ff_dropout = 0.1            # feedforward dropout
        )


        #text = torch.randint(0, 10000, (4, 256))
        #images = torch.randn(4, 3, 256, 256)
        images = np.array([data])
        text = "A mother stands in a kitchen next to the window, she is washing dishes. The sink next to here is overflowing. Her two children are trying to steal from a cookie jar in the top cabinet. A young boy is standing on a stool with his hand in a cookie jar, the stool he is standing on is tipping over. a young girl is watching the boy and reaching for a cookie."
        
        tarr = text.split(' ')
        final = [tarr]
        all_words = list(sorted(frozenset(list(itertools.chain.from_iterable(final)))))
        word_tokens = dict(zip(all_words, range(1, len(all_words) + 1)))
        caption_tokens = [[word_tokens[w] for w in c] for c in final]

        longest_caption = max(len(c) for c in final)
        captions_array = np.zeros((len(caption_tokens), longest_caption), dtype=np.int64)
        for i in range(len(caption_tokens)):
            captions_array[i, :len(caption_tokens[i])] = caption_tokens[i]
            
        captions_array = torch.from_numpy(captions_array).to(self.device)
        captions_mask = captions_array != 0

        self.dalle = DALLE(
            dim = 1024,
            vae = vae,                  # automatically infer (1) image sequence length and (2) number of image tokens
            num_text_tokens = len(word_tokens) + 1,    # vocab size for text
            text_seq_len = longest_caption,         # text sequence length
            depth = 12,                 # should aim to be 64
            heads = 16,                 # attention heads
            dim_head = 64,              # attention head dimension
            attn_dropout = 0.1,         # attention dropout
            ff_dropout = 0.1            # feedforward dropout
        ).to(self.device)

        opt = torch.optim.Adam(self.dalle.parameters(), lr=0.001, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, 0.98)

    def gen_image(self, text):

        tarr = [text]
        all_words = list(sorted(frozenset(list(itertools.chain.from_iterable(tarr)))))
        word_tokens = dict(zip(all_words, range(1, len(all_words) + 1)))
        caption_tokens = [[word_tokens[w] for w in c] for c in tarr]

        images = self.dalle.generate_images(
            tarr,
            cond_scale = 3. # secondly, set this to a value greater than 1 to increase the conditioning beyond average
        )

    def save_model(self):
        pickle.dump(self.dalle, open(self.file, 'wb'))

    def load_model(self):
        self.dalle = pickle.load(open(self.file, "rb"))


if __name__ == "__main__":

    command = ''

    mod = Model()
    while command != ':stop':
        command = input(">")

        if command == ":run":
            mod.train("unknown.png")

        if command == ":save":
            mod.save_model()

        if command == ":load":
            mod.load_model()

        if command == ":gen":
            mod.gen_image("the scene is in the in the kitchen the mother is and the water is running on the floor a child is trying to get a boy is trying to get the little girl is to his falling it to be summer out the window is open the are blowing it must be a gentle breeze theres grass outside in the garden finished certain of the the very tidy gram the mother to have nothing in the house to eat except the look to be almost about the same size perhaps theyre theyre dressed for summer warm weather um you want more the in a short sleeve dress Ill say its warm")