from typing import List, Tuple
import itertools

def gen_fanout_inflation(n: int) -> List[Tuple[int,int]]:
    return [pair for pair in itertools.permutations(range(n), 2) if pair[0] != (pair[1]+1)%n]

def gen_nonfanout_inflation(n: int) -> List[Tuple[int,int]]:
    return [(i,i+1) for i in range(n-1)] + [(n-1,0)]

def gen_fanout_inflation_alt(n: int) -> List[Tuple[int,int]]:
    return list(itertools.permutations(range(n), 2))

if __name__ == "__main__":
    print(gen_fanout_inflation(4))

    print(gen_nonfanout_inflation(5))