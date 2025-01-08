import random 

class Solution(object):
    def merge(self, nums1, m, nums2, n):
        
        n1_i = m - 1
        n2_i = n - 1
        write_i = m + n -1

        for write_i in range(write_i, -1, -1):

            if(n2_i < 0):
                break;
            
            if(n1_i >= 0 and nums1[n1_i] >= nums2[n2_i]):
                nums1[write_i] = nums1[n1_i]
                n1_i -= 1
                write_i -= 1

            else:
                nums1[write_i] = nums2[n2_i]
                n2_i -= 1
                write_i -= 1

    def removeElement(self, nums, val):
        k = len(nums) - 1
        
        for i in range(len(nums) - 1, -1, -1):
            if nums[i] == val:
                nums[i] = nums[k]
                k -= 1
        
        return k + 1

    # def removeDuplicates(self, nums):
        uniqueC = 0

        for i in range(0, len(nums) -1, +1):
                if nums[i] != nums[i+1]:
                    nums[uniqueC] = nums[i]
                    uniqueC += 1
        
        nums[uniqueC] = nums[i+1]
        uniqueC += 1
        
        return uniqueC

    def removeDuplicates(self, nums):
        next_i = 2
            
        for i in range(next_i, len(nums), +1):
            if nums[i] != nums[next_i - 2]:
                nums[next_i] = nums[i]
                next_i += 1
            
        return next_i
    
    def majorityElement(self, nums):
        num_frequency = {}
        for num in nums:
            if num in num_frequency:
                num_frequency[num] = num_frequency[num] + 1
            else:
                num_frequency[num] = 1
        return max(num_frequency, key=num_frequency.get)

    def majorityElement2(self, nums):
        nums.sort()
        return nums[len(nums)//2] 

    def rotate(self, nums, k):
        nums.reverse()
        nums[:k] = reversed(nums[:k])
        nums[k:] = reversed(nums[k:])
    
    def maxProfit(self, prices):
        min_price = prices[0]
        potential_profit = 0

        for price in prices:
            if (price - min_price) > potential_profit:
                potential_profit = price - min_price

            if(price < min_price):
                min_price = price
        
        return potential_profit

    def maxProfit2(self, prices):
        cumulative_profit = 0

        for i in range(0, len(prices)-1, 1):
            if prices[i] < prices[i+1]:
                cumulative_profit += prices[i+1] - prices[i]

        return cumulative_profit

    def canJump(self, nums):
        furthest_index = 0
        for i in range(0, len(nums), 1):
            
            if furthest_index < i:
                return False

            furthest_index = max(furthest_index, i + nums[i])

            if furthest_index >= len(nums)-1:
                return True
    
    def jump(self, nums):
        furthest_index = 0
        number_of_jumps = 0
        end_of_current_jump = 0

        if len(nums) == 1:
            return number_of_jumps

        for i in range(0, len(nums)):
            
            furthest_index = max(furthest_index, i + nums[i]) 

            if i == end_of_current_jump:
                number_of_jumps += 1
                end_of_current_jump = furthest_index

                if end_of_current_jump >= len(nums)-1:
                    break

        return number_of_jumps

    def hIndex(self, citations):
        citations.sort(reverse=True)

        for i in range(0, len(citations)):

            if i + 1 > citations[i]:
                return i
            
        return len(citations)

    def productExceptSelf(self, nums):
        answer = [1]*len(nums)

        prefix_product = 1
        for i in range(0, len(nums)):
            answer[i] = prefix_product
            prefix_product *= nums[i]

        suffix_product = 1
        for i in range(len(nums)-1, -1, -1):
            answer[i] *= suffix_product
            suffix_product *= nums[i]
    
    def canCompleteCircuit(self, gas, cost):
        total_gas = 0
        current_gas = 0
        start_station = 0

        for i in range(0, len(gas)):
            total_gas += gas[i] - cost[i]
            current_gas += gas[i] - cost[i]

            if current_gas < 0:
                start_station = i+1
                current_gas = 0
        
        return start_station if total_gas>=0 else -1

    def romanToInt(self, s):
        roman_map = {
            "I" : 1,
            "V" : 5,
            "X":10,
            "L":50,
            "C":100,
            "D":500,
            "M":1000
            }
        
        value = 0
        i = 0

        while i < len(s):
            if i < len(s)-1 and roman_map[s[i]] < roman_map[s[i+1]]:
                value += roman_map[s[i+1]] - roman_map[s[i]]
                i += 2
            else:
                value += roman_map[s[i]]
                i += 1

        return value

    def intToRoman0(self, num):

        returnRoman = ""

        dict={
            1000: "M",
            500: "D",
            100: "C",
            50: "L",
            10: "X",
            5: "V",
            1: "I"
        }

        thousands, remainder = divmod(num, 1000)
        returnRoman = thousands*"M"

        nineHundread, remainder = divmod(remainder, 900)
        returnRoman += nineHundread*"CM"

        fiveHundread, remainder = divmod(remainder, 500)
        returnRoman += fiveHundread*"D"

        fourHundread, remainder = divmod(remainder, 400)
        returnRoman += fourHundread*"CD"

        hundreads, remainder = divmod(remainder, 100)
        returnRoman += hundreads*"C"

        nintys, remainder = divmod(remainder, 90)
        returnRoman += nintys*"XC"

        fiftys, remainder = divmod(remainder, 50)
        returnRoman += fiftys*"L"

        fourtys, remainder = divmod(remainder, 40)
        returnRoman += fourtys*"XL"

        tens, remainder = divmod(remainder, 10)
        returnRoman += tens*"X"

        nins, remainder = divmod(remainder, 9)
        returnRoman += nins*"IX"

        fivs, remainder = divmod(remainder, 5)
        returnRoman += fivs*"V"

        fours, remainder = divmod(remainder, 4)
        returnRoman += fours*"IV"

        ones, remainder = divmod(remainder, 1)
        returnRoman += ones*"I"

        print(returnRoman)

    def intToRoman(self, num):
        roman_map = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), 
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"), 
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")
    ]
        
        return_String = ""

        for key, roman in roman_map:
           times , num = divmod(num, key)
           return_String += times * roman

        return return_String

    def lengthOfLastWord0(self, s):
        start_index = -1

        for i in range(len(s)-1, 0, -1):

            if start_index == -1 and s[i] == ' ':
                continue

            if start_index == -1 and s[i] != ' ':
                start_index = i
                continue
            
            if s[i] == ' ':
                return start_index - i
                
            if i==0:
                return start_index

    def lengthOfLastWord(self, s):
        s = s.strip()
        return len(s.split()[-1])
        
    def longestCommonPrefix(self, strs):
        prefix = strs[0]

        for i in range(len(prefix)):
            char = prefix[i]

            for word in strs:
                if i >= len(word) or word[i] != char:
                    return prefix[:i]

        return prefix

    def reverseWords1(self, s):
        s_a = s.split()
        s_a = s_a[::-1]
        
        for i in range(len(s_a)-1, -1, -1):
            return_string += s_a[i]
            if i !=0:
                return_string += " "
        
        return return_string

    def reverseWords(self, s):
        s_a = s.split()[::-1]
        
        return " ".join(s_a)
         
    

class Solution2:
    def __init__(self):
        self.memo = []
        self.nums = []

    def canJump_recursive(self, nums):
        self.nums = nums
        self.memo = [-1]*len(nums)
        self.memo[-1] = 1

        return self.canJumpFromPosition_recursive(0, nums)
            
    def canJumpFromPosition_recursive(self, position, nums):
        if self.memo[position] != -1:
            return self.memo[position]==1

        furthest_jump = min(position + nums[position], len(nums)-1 )

        for i in range(position + 1, furthest_jump + 1):
            if self.canJumpFromPosition_recursive(i, nums):
                self.memo[position] = 1
                return True
        
        self.memo[position] = 0
        return False
    
    def canJumpDP(self, nums):

        self.memo = len(nums)*[-1]
        self.memo[-1] = 1

        for i in range(len(nums)-2, -1, -1):

            furthest_position = min(i + nums[i], len(nums)-1)
            
            for j in range(i + 1, furthest_position + 1):
                if self.memo[j] == 1:
                    self.memo[i] = 1
                    break
        
        if self.memo[0] == 1:
            return True
        else: return False



class RandomizedSet(object):

    def __init__(self):
        self.dict={}
        self.values_array = []

    def insert(self, val):
        if val in self.dict:
            return False
        else:
            self.values_array.append(val)
            self.dict[val] = len(self.values_array) - 1
            return True
        
    def remove(self, val):
        if val in self.dict:
            index_to_remove = self.dict[val]
            last_element = self.values_array[-1]
            
            self.values_array[index_to_remove] = last_element            
            self.dict[last_element] = index_to_remove
            self.values_array.pop()
            del self.dict[val]

            return True
        else:
            return False
        

    def getRandom(self):
        return random.choice(self.values_array)
        

if __name__ == "__main__":
    sol = Solution()
    sol2 = Solution2()
    obj = RandomizedSet()

    nums = [3,2,1,0,4]
    nums = [2,3,1,1,4]
    print(sol2.canJumpDP(nums))
    
    # s = "the sky is blue"
    # s = "  hello world  "
    # s = "a good   example"
    # print(sol.reverseWords(s))

    # strs = ["flower","flow","flight"]
    # print(sol.longestCommonPrefix(s))

    #s = "Hello World"
    #s = "   fly me   to   the moon  "
    #s = "luffy is still joyboy"
    #print(sol.lengthOfLastWord(s))
    
    # print(sol.intToRoman(3749))
    
    # s = "MCMXCIV"
    # print(sol.romanToInt(s))
    
    # gas = [1,2,3,4,5]
    # cost = [3,4,5,1,2]
    # print(sol.canCompleteCircuit(gas, cost))
    
    # nums = [1,2,3,4]
    # nums = [-1,1,0,-3,3]
    # nums = [5,6,3,4]
    # print(sol.productExceptSelf(nums))

    # operations = ["RandomizedSet", "insert", "remove", "insert", "getRandom", "remove", "insert", "getRandom"]
    # values = [[], [1], [2], [2], [], [1], [2], []]

    # operations = ["RandomizedSet","insert","remove","insert","getRandom","remove","insert","getRandom"]
    # values = [[],[-1],[-2],[-2],[],[-1],[-2],[]]

    # for op, val in zip(operations, values):
    #     if op == "insert":
    #         result = obj.insert(*val)
    #         print(f"insert({val[0]}) = {result}")
    #         print(obj.values_array)
    #         print(obj.dict)
    #     elif op == "remove":
    #         result = obj.remove(*val)
    #         print(f"remove({val[0]}) = {result}")
    #         print(obj.values_array)
    #         print(obj.dict)
    #     elif op == "getRandom":
    #         result = obj.getRandom()
    #         print(f"getRandom() = {result}")
    #         print(obj.values_array)
    #         print(obj.dict)
    
    
    # citations = [3,0,6,1,5]
    # citations = [1,3,1]
    # print(sol.hIndex(citations))

    # nums = [0]
    # print(sol.jump(nums))
    
    # nums = [3,2,1,0,4]
    # print(sol.canJump(nums))
    
    # prices = [7,1,5,3,6,4]
    # print(sol.maxProfit2(prices))
    
    # nums = [1,2]
    # k = 3
    # value = sol.rotate(nums,k)
    # print(nums)
    
    # nums = [3]
    # value = sol.majorityElement2(nums)
    # print(value)
    
    # nums = [2,2,1,1,1,2,2]
    # value = sol.majorityElement(nums)
    # print(value)
    
    # nums = [0,0,1,1,1,1,2,3,3]
    # print(sol.removeDuplicates(nums))
    # print(nums)
    
    # nums = [0,0,1,1,1,2,2,3,3,4]
    # print(sol.removeDuplicates(nums))
    # print(nums)
    # nums1 = [1], m = 1, nums2 = [2,5,6], n = 3
    # sol.merge(nums1, m, nums2, n)
    # print(nums1)
    # nums = [3,2,2,3]
    # val = 3
    # nums = [0,1,2,2,3,0,4,2]
    # val = 2
    # print(sol.removeElement(nums, val))
