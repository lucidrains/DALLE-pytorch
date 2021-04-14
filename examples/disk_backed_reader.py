import os
import numpy as np
from tqdm.autonotebook import trange, tqdm
import hashlib


def int32_to_bytes(n: int) -> bytes:
    return n.to_bytes(4, byteorder='big', signed=True)


def bytes_to_int32(bb: bytes) -> int:
    assert len(bb) == 4
    return int.from_bytes(bb, byteorder='big', signed=True)


def int64_to_bytes(n: int) -> bytes:
    return n.to_bytes(8, byteorder='big', signed=True)


def bytes_to_int64(bb: bytes) -> int:
    assert len(bb) == 8
    return int.from_bytes(bb, byteorder='big', signed=True)

            
class DiskBackedListReader:
    def __init__(self, file_name, max_record_size=100000000):
        self.max_record_size = max_record_size
        self.file_name = file_name
        
        self.records_file = open(self.file_name, "rb")
        offsets_file_name = self.file_name + ".offsets"
        self.offsets_file = open(offsets_file_name, "rb")
        
        offsets_file_size = os.path.getsize(offsets_file_name)
        assert offsets_file_size % 8 == 0
        self.num_records = offsets_file_size // 8
        
        self.records_file_size = os.path.getsize(self.file_name)
        self.at_record = 0
        
    def __len__(self):
        return self.num_records
        
    def next_item(self) -> bytes:
        record_size = bytes_to_int32(self.records_file.read(4))
        assert 0 < record_size <= self.max_record_size
        assert self.records_file_size - self.records_file.tell() >= record_size
        self.at_record += 1
        return self.records_file.read(record_size)
    
    def move_to(self, idx: int):
        assert 0 <= idx < self.num_records
        if idx == self.at_record:
            return
        
        self.offsets_file.seek(idx * 8, os.SEEK_SET)
        offset = bytes_to_int64(self.offsets_file.read(8))
        self.records_file.seek(offset, os.SEEK_SET)        
        self.at_record = idx
        
    def __getitem__(self, idx: int) -> bytes:
        self.move_to(idx)
        return self.next_item()
    
    def close(self):
        self.records_file.close()
        self.offsets_file.close()
        
    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()    
                    

class DiskBackedListWriter:
    def __init__(self, file_name, max_record_size=100000000):
        self.max_record_size = max_record_size
        assert not os.path.exists(file_name), "Will not overwrite existing DB"
        self.file = open(file_name, "wb")
        offsets_file_name = file_name + ".offsets"
        self.offsets_file = open(offsets_file_name, "wb")
        self.file_end_pos = 0
        self.num_records = 0
        self.entered = False
        
    def __len__(self):
        return self.num_records
        
    def add(self, record: bytes):
        assert self.entered, "To avoid write errors, writing to DiskBackedList is only permitted inside with: blocks"
        assert 0 < len(record) <= self.max_record_size
        self.file.seek(0, os.SEEK_END)
        assert self.file.tell() == self.file_end_pos, "DB file is damaged, most likely due to concurrent writes"
        self.offsets_file.seek(0, os.SEEK_END)
        assert self.offsets_file.tell() == self.num_records * 8, "DB file is damaged, most likely due to concurrent writes"        
        self.file.write(int32_to_bytes(len(record)))
        self.file.write(record)
        self.offsets_file.write(int64_to_bytes(self.file_end_pos))
        self.file_end_pos += len(record) + 4           
        self.num_records += 1        
    
    def close(self):
        self.file.close()
        self.offsets_file.close()
        self.entered = False
        
    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, type, value, tb):
        self.close()
        
            
def decode_cc(cc_data):
    source_length = bytes_to_int32(cc_data[:4])
    assert 0 < source_length < len(cc_data) - 4, source_length
    src = cc_data[4:4 + source_length].decode("utf-8")
    caption = cc_data[4 + source_length:].decode("utf-8")
    return src, caption
    

def deserialize_image_captions(img_caps_data: bytes):
    assert len(img_caps_data) > 4
    num_cc = bytes_to_int32(img_caps_data[:4])
    img_caps_data = img_caps_data[4:]
    caps = []
    for _ in range(num_cc):
        assert len(img_caps_data) > 4        
        cc_len = bytes_to_int32(img_caps_data[:4])
        caps.append(decode_cc(img_caps_data[4:cc_len + 4]))
        img_caps_data = img_caps_data[cc_len + 4:]
        
    return img_caps_data, caps

                
def checksum(db_file):
    md5_sum = 0
    with DiskBackedListReader(db_file) as db:
        for _ in trange(len(db)):
            md5_sum ^= int.from_bytes(hashlib.md5(db.next_item()).digest(), byteorder="big")
            
    return md5_sum


previews_checksum = 5541702935356995129647168840502502203


def fast_shuffle(src_db, dest_db, tmp_db, seed, chunk_records=1000, chunks_in_batch=10000):    
    with DiskBackedListReader(src_db) as sdb:
        with DiskBackedListWriter(dest_db) as ddb:
            num_chunks = (len(sdb) - 1) // chunk_records + 1
            num_batches = (num_chunks - 1) // chunks_in_batch + 1
            np.random.seed(seed)
            chunks_order = np.arange(num_chunks)
            np.random.shuffle(chunks_order)
            for bi in range(num_batches):
                batch_chunks = chunks_order[bi * chunks_in_batch:(bi + 1) * chunks_in_batch]
                with DiskBackedListWriter(tmp_db) as tdb:
                    for ci in tqdm(batch_chunks, desc=f"{bi + 1}/{num_batches} src->tmp"):
                        for i in range(ci * chunk_records, min(len(sdb), (ci + 1) * chunk_records)):
                            tdb.add(sdb[i])
                            
                with DiskBackedListReader(tmp_db) as tdb:
                    batch_order = np.arange(len(tdb))
                    np.random.shuffle(batch_order)
                    for ti in tqdm(batch_order, desc=f"{bi + 1}/{num_batches} tmp->dst"):
                        ddb.add(tdb[ti])
                        
                os.remove(tmp_db)
                os.remove(tmp_db + ".offsets")                        
                        
            assert len(sdb) == len(ddb)
            
    assert os.path.getsize(src_db) == os.path.getsize(dest_db)
                    
