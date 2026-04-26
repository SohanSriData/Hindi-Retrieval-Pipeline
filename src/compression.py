from typing import List 

def gap_encode(numbers: List[int]) -> List[int]:
    if not numbers:
        return []
    gaps = [numbers[0]]
    for i in range(1, len(numbers)):
        gaps.append(numbers[i] - numbers[i-1])
    return gaps

def gap_decode(gaps: List[int]) -> List[int]:
    if not gaps:
        return []
    nums = [gaps[0]]
    
    for g in gaps[1:]:
        nums.append(nums[-1] + g)
    return nums

def vb_encode_number(n: int) -> bytes:
    bytes_list = []
    
    # Extract 7-bit chunks
    while True:
        bytes_list.insert(0, n % 128)
        if n < 128:
            break
        n //= 128
        
    bytes_list[-1] += 128
    
    return bytes(bytes_list)

def vb_encode_list(nums: List[int]) -> bytes:
    b = bytearray()
    for n in nums:
        b.extend(vb_encode_number(n))
    return bytes(b)

def vb_decode_stream(b: bytes) -> List[int]:
    numbers = []
    n = 0
    
    for byte in b:
        if byte < 128:
            # MSB = 0: Number continues in the next byte
            n = 128 * n + byte
        else:
            # MSB is 1: This is the last byte of number
            n = 128 * n + (byte - 128)
            numbers.append(n)
            n = 0
            
    return numbers

def compress_docids(doc_ids: List[int]) -> bytes:
    if not doc_ids:
        return b''
    gaps = gap_encode(doc_ids)
    return vb_encode_list(gaps)

def decompress_docids(b_stream: bytes) -> List[int]:
    if not b_stream:
        return []
    gaps = vb_decode_stream(b_stream)
    return gap_decode(gaps)


if __name__ == '__main__':
    ids = [1, 130, 25000, 25005] 
    
    compressed_bytes = compress_docids(ids)
    print(f'Encoded Bytes: {compressed_bytes}')
    print(f'Decoded IDs: {decompress_docids(compressed_bytes)}')