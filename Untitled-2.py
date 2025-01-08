class Solution(object):
    def merge(self, nums1, m, nums2, n):
        # Initialize indices
        n1_i = m - 1
        n2_i = n - 1
        write_i = m + n - 1

        # Merge nums1 and nums2 from the end
        for write_i in range(write_i, -1, -1):
            if n2_i < 0:
                break

            if n1_i >= 0 and nums1[n1_i] > nums2[n2_i]:
                nums1[write_i] = nums1[n1_i]
                n1_i -= 1
            else:
                nums1[write_i] = nums2[n2_i]
                n2_i -= 1


# Test the Solution
if __name__ == "__main__":
    nums1 = [1, 2, 3, 0, 0, 0]  # Array 1 (with space for nums2)
    m = 3  # Number of elements in nums1
    nums2 = [2, 5, 6]  # Array 2
    n = 3  # Number of elements in nums2

    # Create Solution object and call merge
    sol = Solution()
    sol.merge(nums1, m, nums2, n)

    # Print the merged array
    print(nums1)
