from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        
        chars = [c for w in corpus for c in w ]
        merges = []

        for m in range(num_merges):
            pairs = {}
            for i in range(len(chars) - 1):
                k = (chars[i], chars[i+1])
                if k not in pairs:
                    pairs[k] = 1
                else:
                    pairs[k] += 1
            pairs = dict(sorted(pairs.items(), key=lambda item: item[1], reverse=True))

            sorting = [key for key, value in pairs.items() if value == pairs[next(iter(pairs))]]
            sorting.sort()

            new_chars = []
            i = 0
            while i<len(chars):
                if i == len(chars)-1:
                    new_chars.append(chars[i])
                    break
                elif (chars[i], chars[i+1])==sorting[0]:
                    new_chars.append(chars[i]+chars[i+1])
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            chars = new_chars
            merges.append(sorting[0])

        return merges

    

        

            
            
            
    