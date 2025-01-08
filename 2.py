from typing import Optional
from collections import deque
import math

class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next

class solutionTriangle_memoize:
    def __init__(self):
        self.triangle = []
        self.memo = []

    def minimumTotal(self, triangle):
        self.triangle = triangle
        self.memo = [[-1] * len(row) for row in triangle]
        self.memo[-1] = self.triangle[-1][:]
        
        value_to_return = self.minimumTotal_Position(0,0)
        
        return value_to_return
    

    def minimumTotal_Position(self, row, position):
        if(self.memo[row][position] == -1):

            if row == len(self.triangle) -1:
                return self.triangle[row][position]

            value = self.triangle[row][position] + min(self.minimumTotal_Position(row+1 , position), self.minimumTotal_Position(row+1 , position + 1))
            self.memo[row][position]= value
            return value

        else:
            return self.memo[row][position]

class solutionTriangle_DP:
    def __init__(self):
        self.triangle = []
    
    def minimumTotal(self, triangle):
        self.triangle = triangle 

        for i in range(len(self.triangle)-2, -1, -1):
            for j in range(0, len(self.triangle[i])):
                self.triangle[i][j] = self.triangle[i][j] + min(self.triangle[i+1][j], self.triangle[i+1][j+1])
        
        return self.triangle[0][0]

class SolutionminPathSum(object):
    def minPathSum(self, grid):
        m = len(grid)
        n = len(grid[0])

        dp_grid = [[0 for _ in range(n)] for _ in range(m)]

        for row in range(m-1, -1, -1):
            for column in range(n-1, -1, -1):

                if row == len(grid)-1 and column == len(grid[0])-1:
                    dp_grid[row][column] = grid[row][column]
                    continue

                if row < len(grid)-1 and column<len(grid[0])-1:
                    dp_grid[row][column] = grid[row][column] + min(dp_grid[row][column+1], dp_grid[row+1][column])
                    continue
                
                if row < len(grid)-1:
                    dp_grid[row][column] = grid[row][column] + dp_grid[row+1][column]
                    continue

                if column < len(grid[0])-1:
                    dp_grid[row][column] = grid[row][column] + dp_grid[row][column+1]
                    continue

        
        return dp_grid[0][0]
    
    def moveable(self, row, column, grid):

        if row == len(grid)-1 and column == len(grid[0])-1:
            return "none"

        if row < len(grid)-1 and column<len(grid[0])-1:
            return "both"
        
        if row < len(grid)-1:
            return "down"

        if column < len(grid[0])-1:
            return "right"

class uniquePathsWithObstacles(object):
    # def uniquePathsWithObstacles1(self, obstacleGrid):
    #     rows_count = len(obstacleGrid)
    #     columns_count = len(obstacleGrid[0])
        
    #     if rows_count == 1 and columns_count == 1:
    #         return 1 if obstacleGrid[0][0] == 0 else 0

    #     dp_grid = [[0 for _ in range(columns_count)] for _ in range(rows_count)]
        
    #     #SET -2 FOR TARGET
    #     if obstacleGrid[rows_count-1][columns_count-1] == 1:
    #         return 0
    #     else:
    #         dp_grid[rows_count-1][columns_count-1] = -2

    #     #FILL LAST COLUMN
    #     for row in range(rows_count - 2, -1, -1):
            
    #         #SET -1 FOR OBSTACLE
    #         if obstacleGrid[row][columns_count-1] == 1:
    #             dp_grid[row][columns_count-1] = -1
    #             continue
            
    #         #SET 0 OR -1
    #         if obstacleGrid[row][columns_count-1] == 0:

    #             #SET -1 OR 1 DEPENDING ON NEXT
    #             if dp_grid[row+1][columns_count-1] == -1:
    #                 dp_grid[row][columns_count-1] = -1
    #             else:
    #                 dp_grid[row][columns_count-1] = 1
                    
    #     #FILL LAST ROW
    #     for column in range(columns_count - 2, -1, -1):
    #         if obstacleGrid[rows_count-1][column] == 1:
    #             dp_grid[rows_count-1][column] = -1
    #             continue
            
    #         if dp_grid[rows_count-1][column+1] == -1:
    #             dp_grid[rows_count-1][column] = -1
    #         else:
    #             dp_grid[rows_count-1][column] = 1

    #     print(dp_grid)

    #     for column in range(columns_count -2, -1, -1):
    #         for row in range(rows_count -2, -1, -1):
    #             right = dp_grid[row][column+1]
    #             under = dp_grid[row+1][column]

    #             if obstacleGrid[row][column] == 1:
    #                 dp_grid[row][column] = -1
    #                 continue

    #             if right == -1 and under == -1:
    #                 dp_grid[row][column] = -1
                
    #             elif right == -1 and under != -1:
    #                 dp_grid[row][column] = under
                
    #             elif right != -1 and under == -1:
    #                 dp_grid[row][column] = right
                
    #             else: dp_grid[row][column] = right + under
        
    #     print(dp_grid)

    #     return 0 if dp_grid[0][0] == -1 else dp_grid[0][0]
    
    def uniquePathsWithObstacles(self, obstacleGrid):
        rows_count = len(obstacleGrid)
        columns_count = len(obstacleGrid[0])

        dp_grid = [[0 for _ in range(columns_count)] for _ in range(rows_count)]
        
        #SET -2 FOR TARGET
        if obstacleGrid[rows_count-1][columns_count-1] == 0:
            dp_grid[rows_count-1][columns_count-1] = 1

        #FILL LAST COLUMN
        for row in range(rows_count - 2, -1, -1):
            if obstacleGrid[row][columns_count - 1] == 0:
                dp_grid[row][columns_count - 1] = dp_grid[row+1][columns_count-1]

        #Fill Last Row
        for column in range(columns_count-2, -1, -1):
            if obstacleGrid[rows_count-1][column] == 0:
                dp_grid[rows_count-1][column] = dp_grid[rows_count-1][column+1]

        for column in range(columns_count -2, -1, -1):
            for row in range(rows_count -2, -1, -1):
                if obstacleGrid[row][column] == 0:
                    dp_grid[row][column] = dp_grid[row+1][column] + dp_grid[row][column+1]

        return dp_grid[0][0]

    def uniquePathsWithObstacles0(self, obstacleGrid):
        rows_count = len(obstacleGrid)
        columns_count = len(obstacleGrid[0])
        
        # Initialize DP grid
        dp_grid = [[0 for _ in range(columns_count)] for _ in range(rows_count)]
        
        # Set the target cell
        if obstacleGrid[rows_count - 1][columns_count - 1] == 0:
            dp_grid[rows_count - 1][columns_count - 1] = 1
        
        # Fill the last row
        for column in range(columns_count - 2, -1, -1):
            if obstacleGrid[rows_count - 1][column] == 0:
                dp_grid[rows_count - 1][column] = dp_grid[rows_count - 1][column + 1]
        
        # Fill the last column
        for row in range(rows_count - 2, -1, -1):
            if obstacleGrid[row][columns_count - 1] == 0:
                dp_grid[row][columns_count - 1] = dp_grid[row + 1][columns_count - 1]
        
        print(obstacleGrid)
        print(dp_grid)

        # Fill the rest of the grid
        for row in range(rows_count - 2, -1, -1):
            for column in range(columns_count - 2, -1, -1):
                if obstacleGrid[row][column] == 0:
                    dp_grid[row][column] = dp_grid[row + 1][column] + dp_grid[row][column + 1]
        
        # Return the result from the top-left corner
        return dp_grid[0][0]

class longestPalindrome():
    def longestPalindrome(self, s):
        start = 0
        palindrome_length = 0

        n = len(s)
        dp_grid = [[False]*n for _ in range(n)]
        #length = 1
        for i in range(n):
            dp_grid[i][i] = True
            start = i
            palindrome_length = 1
        
        #length = 2
        for i in range(n-1):
            if s[i] == s[i+1]:
                dp_grid[i][i+1]=True
                start = i
                palindrome_length = 2
        
        print(dp_grid)

        #length > 2
        for length in range(3, n+1):
            for start_index in range(0, n-length+1):
                end_index = start_index + length -1 

                if s[start_index] == s[end_index] and dp_grid[start_index+1][end_index-1] == True:
                    dp_grid[start_index][end_index] = True
                    start = start_index
                    palindrome_length = length

        return s[start:start+palindrome_length]

class sol_isInterleave_memoize():
    
    def __init__(self):
        self.S1 = ""
        self.S2 = ""
        self.S3 = ""
        self.memo = {}

    def isInterleave(self, s1, s2, s3):
        self.S1 = s1
        self.S2 = s2
        self.S3 = s3
        
        return(self.isInterleave_rec(0,0,0))
    
    def isInterleave_rec(self, s1_index, s2_index, s3_index):
        
        if s1_index == len(self.S1) and s2_index == len(self.S2) and s3_index == len(self.S3):
            return True
        
        if s3_index == len(self.S3):
            return False

        if (s1_index, s2_index) in self.memo:
            return self.memo[(s1_index, s2_index)]

        if s1_index <= len(self.S1)-1 and self.S3[s3_index] == self.S1[s1_index]:
            if self.isInterleave_rec(s1_index+1, s2_index, s3_index+1):
                self.memo[(s1_index, s2_index)] = True
                return True
                
        if s2_index <= len(self.S2)-1 and self.S3[s3_index] == self.S2[s2_index]:
            if self.isInterleave_rec(s1_index, s2_index+1, s3_index+1):
                self.memo[(s1_index, s2_index)] = True
                return True
                
        return False

class sol_isInterleave_dp():
    
    def isInterleave(self, s1, s2, s3):
        if len(s1) + len(s2) != len(s3):
            return False
        
        dp_grid = [[False for _ in range(len(s2) + 1)] for _ in range(len(s1) + 1)]
        dp_grid[0][0] = True

        for i in range(1, len(s2)+1):
            if s2[i-1] == s3[i-1] and dp_grid[0][i-1] == True:
                dp_grid[0][i] = True
        
        for i in range(1, len(s1)+1):
            if s1[i-1] == s3[i-1] and dp_grid[i-1][0] == True:
                dp_grid[i][0] = True

        for i in range(1, len(s1)+1): #1
            for j in range(1, len(s2)+1): #2
                
                if s1[i-1] == s3[i+j-1] and dp_grid[i-1][j] == True:
                    dp_grid[i][j] = True
                
                elif s2[j-1] == s3[i+j-1] and dp_grid[i][j-1] == True:
                    dp_grid[i][j] = True

        return dp_grid[len(s1)][len(s2)]

class minDistance_dp():
    def minDistance(self, word1, word2): 
        dp_grid = [[0 for _ in range (len(word2) + 1)] for _ in range(len(word1) + 1)]

        for i in range(1, len(word2) + 1):
            dp_grid[0][i] = i
        
        for i in range(1, len(word1) + 1):
            dp_grid[i][0] = i
        
        for i in range(1, len(word1) + 1):
            for j in range(1, len(word2) + 1):
                if word1[i-1] == word2[j-1]:
                    dp_grid[i][j] = dp_grid[i-1][j-1]
                
                else:
                    insert = 1 + dp_grid[i][j-1]
                    delete = 1 + dp_grid[i-1][j]
                    replace = 1 + dp_grid[i-1][j-1]
                    
                    dp_grid[i][j] = min(insert, delete, replace)

        return dp_grid[len(word1)][len(word2)]

class Solution0:
    def __init__(self):
        self.Inorder = []
    
    def inorder(self, root):
        if not root:
            return
        self.inorder(root.left)
        self.Inorder.append(root.val)
        self.inorder(root.right)
    
    def minDiffInBST(self, root):
        if not root:
            return 0
        self.inorder(root)
        res = float('inf')
        for i in range(1, len(self.Inorder)):
            res = min(res, self.Inorder[i] - self.Inorder[i-1])
        return res

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def build_tree(values):
        if not values:
            return None
        
        root = TreeNode(values[0])
        queue = deque([root])
        i = 1
        
        while queue and i < len(values):
            node = queue.popleft()
            
            if values[i] is not None:
                node.left = TreeNode(values[i])
                queue.append(node.left)
            i += 1
            
            if i < len(values) and values[i] is not None:
                node.right = TreeNode(values[i])
                queue.append(node.right)
            i += 1
            
        return root

class Solution_BST_MinDiff:
    def getMinimumDifference(self, root):
        node_values_inorder = []
        minimum = 1e9

        def inorder(node):
            if node is None:
                return

            inorder(node.left)
            node_values_inorder.append(node.val)
            inorder(node.right)

        inorder(root)
    
        for i in range(1, len(node_values_inorder)):
            minimum = min(minimum, (node_values_inorder[i] - node_values_inorder[i-1]))
        
        return minimum
    
class Solution_BST_Kth_Smallest(object):
    def kthSmallest(self, root, k):
        state = [0, None]

        def inorder(node):
            if node is None:
                return

            inorder(node.left)
            state[0] += 1
            if state[0] == k:
                state[1] = node.val
                return
            inorder(node.right)

        inorder(root)
        
        return state[1]

class Solution_isValidBST(object):
    def isValidBST(self, root):

        def validate(node, min = -math.inf, max = math.inf):

            if node is None:
                return True
            
            if  node.val <= min or node.val >= max:
                return False
            
            return validate(node.left, min, node.val) and validate(node.right, node.val, max)
        
        return (validate(root))
        
class Solution_rightSideView(object):
    
    def rightSideView(self, root):
        if not root:
            return []
        
        queue = deque([root])
        result = []

        while queue:
            queue_length = len(queue)

            for i in range(0, queue_length):
                
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                
                if node.right:
                    queue.append(node.right)
                
                if i == queue_length - 1:
                    result.append(node.val)
            
        return result

class Solution_averageOfLevels(object):
    def averageOfLevels(self, root):
        
        queue = deque([root])
        answer = []

        while queue:
            level_length = len(queue)
            level_sum = 0

            for i in range(level_length):
                
                node = queue.popleft()
                level_sum += node.val

                if node.left:
                    queue.append(node.left)

                if node.right:
                    queue.append(node.right)

                if i == level_length - 1:
                    answer.append(round(level_sum / level_length, 5))

        return answer

class Solution_levelOrder(object):

    def levelOrder(self, root):
        
        if root is None:
            return []
        
        queue = deque([root])
        answer = [] 

        while queue:
            level_length = len(queue)
            level_answer = []

            for i in range(level_length):
                node = queue.popleft()

                if node.left:
                    queue.append(node.left)
                
                if node.right:
                    queue.append(node.right)
                
                level_answer.append(node.val)
                
                if i == level_length - 1:
                    answer.append(level_answer)
    
        return answer

class Solution_zigzagLevelOrder():
    def zigzagLevelOrder(self, root):
        
        if root is None:
            return []
        
        queue = deque([root])
        answer = []
        level_reverse = False

        while queue:
            level_length = len(queue)
            level_answer = []

            for i in range(level_length):
                node = queue.popleft()
                level_answer.append(node.val)

                if node.left:
                    queue.append(node.left)

                if node.right:
                    queue.append(node.right)

                if i == level_length - 1:
                    if level_reverse:
                        level_answer.reverse()

                    answer.append(level_answer)
                    level_reverse = not level_reverse
        
        return answer

class Solution_maxDepth(object):
    def maxDepth(self, root):

        if root is None:
            return []

        queue = deque([root])
        level = 0
        
        while queue:
            level += 1
            level_length = len(queue)

            for _ in range(level_length):
                node = queue.popleft()

                if node.left:
                    queue.append(node.left)
                
                if node.right:
                    queue.append(node.right)
        
        return level

class Solution_isSameTree0(object):
    def isSameTree(self, p, q):
        if self.DFS_Traverse(p) == self.DFS_Traverse(q):
            return True
        else:
            return False

    #changed to another approach
    def DFS_Traverse(self, root):
        if root is None:
            return []
        queue = deque([root])
        answer = []

        while queue:
            level_length = len(queue)
            for _ in range(level_length):
                
                node = queue.popleft()
                if node != []:
                    answer.append(node.val)
                    if node.left:
                        queue.append([node.left])
                    else:
                        queue.append([None])
                    
                    if node.right:
                        queue.append(node.right)
                    else:
                        queue.append([None])
                
                else:
                    answer.append([None])

        return answer
    
class Solution_isSameTree(object):
    def isSameTree(self, p, q):
        
        if p is None and q is None:
            return True
        
        if p is None or q is None or p.val != q.val:
            return False
        
        if self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right):
            return True
        else:
            return False

class Solution_invertTree(object):
    def invertTree(self, root):
        if root is None:
            return None

        self.invertTree(root.left)
        self.invertTree(root.right)

        temp = root.left
        root.left  = root.right
        root.right = temp

        return root

class Solution_isSymmetric(object):
    def isSymmetric(self, root):
        if root is None:
            return True

        def two_trees_mirror(root_1, root_2):
            
            if root_1 is None and root_2 is None:
                return True
            
            if root_1 is None or root_2 is None or root_1.val != root_2.val:
                return False

            if two_trees_mirror(root_1.left, root_2.right) and two_trees_mirror(root_1.right, root_2.left):
                return True
            
            else:
                return False

        return two_trees_mirror(root.left, root.right)

class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right    

class Solution_buildTree(object):
    def buildTree0(self, preorder, inorder):
        
        if not preorder or not inorder:
            return

        root_value = preorder[0]
        root = TreeNode(root_value)

        root_position = inorder.index(root.val)

        left = self.buildTree(preorder[1:root_position + 1], inorder[:root_position])
        right = self.buildTree(preorder[root_position + 1:], inorder[root_position + 1:])

        root.left = left
        root.right = right

        return root
    
    def buildTree(self, preorder, inorder):
        
        inorder_index_map = {value: index for index, value in enumerate(inorder)}
        self.preorder_index = 0
        

        def recursive(left, right):
            if left > right:
                return

            root_value = preorder[self.preorder_index]
            root = TreeNode(root_value)

            self.preorder_index += 1

            root.left = recursive(left, inorder_index_map[root_value] - 1)
            root.right = recursive(inorder_index_map[root_value] + 1, right)

            return root
        
        return recursive(0, len(preorder)-1)

class Solution_hasPathSum(object):
    def hasPathSum(self, root, targetSum):

        if root is None:
            return False

        if root.left is None and root.right is None and targetSum - root.val == 0:
            return True
        
        if root.left:
            if self.hasPathSum(root.left, targetSum - root.val):
                return True
        
        if root.right:
            if self.hasPathSum(root.right, targetSum - root.val):
                return True            
        
        return False
        
class Solution_buildTree_post(object):
    def buildTree(self, inorder, postorder):

        self.postorder_index = len(postorder) - 1
        inorder_map = {val: idx for idx, val in enumerate(inorder)}

        def recursive(left, right):

            if right < left:
                return None

            root_value = postorder[self.postorder_index]
            # root_value = postorder.pop()
            root_index_inorder = inorder_map[root_value]

            self.postorder_index -= 1
            root = TreeNode(root_value)

            root.right = recursive(root_index_inorder + 1, right)
            root.left = recursive(left, root_index_inorder - 1)

            return root
        
        return recursive(0, len(inorder)-1)

class Solution_countNodes(object):
    
    def countNodes(self, root):
        if not root:
            return 0

        def count_nodes(node):
            height = 0
            while node:
                height += 1
                node = node.left
            return height

        left_count = count_nodes(root.left)
        right_count = count_nodes(root.right)

        if left_count == right_count:
            return (1 << left_count) + self.countNodes(root.right)
        
        else:
            return (1 << right_count) + self.countNodes(root.left)

class Solution_sumNumbers(object):
    
    def sumNumbers(self, root):
        if root is None:
            return 0

        def recursive(node, value):
            sum = 0
            if node.left is None and node.right is None:
                return int(value*10 + node.val)
            
            if node.left:
                sum += recursive(node.left, value*10 + node.val)

            if node.right:
                sum += recursive(node.right, value*10 + node.val)

            return sum

        return recursive(root, 0)


    # def sumNumbers(self, root):
    #     if root is None:
    #         return 0

    #     def recursive(node, value=""):
    #         sum = 0
    #         if node.left is None and node.right is None:
    #             return int(value + str(node.val))
            
    #         if node.left:
    #             sum += recursive(node.left, value + str(node.val))

    #         if node.right:
    #             sum += recursive(node.right, value + str(node.val))

    #         return sum

    #     return recursive(root, "")

class Solution_connect0(object):
    def connect(self, root):
        queue = deque([root])
        answer = []

        while queue:
            queue_length = len(queue)
            for i in range(queue_length):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                
                if node.right:
                    queue.append(node.right)

                answer.append(node.val)

                if i == queue_length - 1:
                    answer.append('#')
        
        return answer

class Solution_connect(object):
    def connect(self, root):
        
        current = root

        while current:
            dummy = Node(0)
            tail = dummy

            while current:

                if current.left:
                    tail.next = current.left
                    tail = tail.next
                
                if current.right:
                    tail.next = current.right
                    tail = tail.next
                
                current = current.next
            
            current = dummy.next
        
        return root
        
        





if __name__ == "__main__":

    values = [1,2,3,4,5,None,7]
    root = build_tree(values)

    sol = Solution_connect0()
    print(sol.connect(root))

    # values = [1,2,3]
    # values = [4,9,0,5,1]
    # root = build_tree(values)
    # sol = Solution_sumNumbers()
    # print(sol.sumNumbers(root))
    
    # values = [1,2,3,4,5]
    # root = build_tree(values)
    # sol = Solution_countNodes()
    # print(sol.countNodes(root))

    # inorder = [9,3,15,20,7]
    # postorder = [9,15,7,20,3]

    # sol = Solution_buildTree_post()

    # sol2 = Solution_levelOrder()
    # print(sol2.levelOrder(sol.buildTree(inorder, postorder)))

    # values = [5, 4, 8, 11, None, 13, 4, 7, 2, None, None, None, 1]
    # targetSum = 22
    # root = build_tree(values)
    
    # sol = Solution_hasPathSum()
    # print(sol.hasPathSum(root, targetSum))

    # preorder = [3,9,20,15,7]
    # inorder = [9,3,15,20,7]

    # sol = Solution_buildTree()
    # sol2 = Solution_levelOrder()
    
    # print(sol2.levelOrder(sol.buildTree0(preorder, inorder)))
    
    # values = [1,2,2,3,4,4,3]
    # values = [1,2,2,None,3,None,3]
    # root = build_tree(values)
    
    # sol = Solution_isSymmetric()
    # print(sol.isSymmetric(root))
    
    # values = [4,2,7,1,3,6,9]
    # root = build_tree(values)
    
    # sol = Solution_invertTree()
    # sol2 = Solution_levelOrder()

    # print(sol2.levelOrder(sol.invertTree(root)))
    
    # p = [1,2,3]
    # q = [1,2,3]
    # p_tree = build_tree(p)
    # q_tree = build_tree(q)
    # sol = Solution_isSameTree()
    # print(sol.isSameTree(p_tree, q_tree))

    # values = [3,9,20,None,None,15,7]
    # values = [1,None,2]
    # values = []
    # root = build_tree(values)
    # sol = Solution_maxDepth()
    # print(sol.maxDepth(root))
    
    # values = [3,9,20,None,None,15,7]
    # values = [1,2,3,4,None,None,5]
    # root = build_tree(values)
    # sol = Solution_zigzagLevelOrder()
    # print(sol.zigzagLevelOrder(root))
    
    # values = [3,9,20,None,None,15,7]
    # values = [3,9,20,15,7]
    # root = build_tree(values)
    # sol = Solution_averageOfLevels()
    # print(sol.averageOfLevels(root))
    
    # values = []
    # root = build_tree(values)
    # sol = Solution_rightSideView()
    # print(sol.rightSideView(root))

    # values = [5,1,4,None,None,3,6]
    # values = [2,1,3]
    # root = build_tree(values)
    
    # sol = Solution_isValidBST()
    # print(sol.isValidBST(root))

    # values = [3, 1, 4, None, 2]
    # k = 1
    # root = build_tree(values)
    # solution = Solution_BST_Kth_Smallest()
    # print(solution.kthSmallest(root, k))
    
    # values = [4, 2, 6, 1, 3]
    # root = build_tree(values)
    # solution = Solution_BST_MinDiff()
    # print(solution.getMinimumDifference(root))  # Output: 1

    # word1 = "horse"
    # word2 = "ros"
    # sol = minDistance_dp()
    # print(sol.minDistance(word1, word2))

    # s1 = "bbbbbabbbbabaababaaaabbababbaaabbabbaaabaaaaababbbababbbbbabbbbababbabaabababbbaabababababbbaaababaa"
    # s2 = "babaaaabbababbbabbbbaabaabbaabbbbaabaaabaababaaaabaaabbaaabaaaabaabaabbbbbbbbbbbabaaabbababbabbabaab"
    # s3 = "babbbabbbaaabbababbbbababaabbabaabaaabbbbabbbaaabbbaaaaabbbbaabbaaabababbaaaaaabababbababaababbababbbababbbbaaaabaabbabbaaaaabbabbaaaabbbaabaaabaababaababbaaabbbbbabbbbaabbabaabbbbabaaabbababbabbabbab"
    # s1 = "aab"
    # s2 = "axy"
    # s3 = "aaxaby"

    # sol = sol_isInterleave_dp()
    # print(sol.isInterleave(s1,s2,s3))

    # longestPalindrome = longestPalindrome()
    # s = "babad"
    # #s = "cbbd"
    # s = "a"
    # s = "ccc"
    # print(longestPalindrome.longestPalindrome(s))

    # sol = solutionTriangle_memoize()
    # solDP = solutionTriangle_DP()
    # solPathSum = SolutionminPathSum()
    # pathsWithObstacles = uniquePathsWithObstacles()

    # obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
    # obstacleGrid = [[0,1],[0,0]]
    # obstacleGrid = [[1,0],[0,0]]
    # #obstacleGrid = [[0,0],[1,0]]
    # #obstacleGrid = [[0,1,0],[0,1,0],[0,0,0]]
    # #obstacleGrid = [[1,1,1],[1,1,1],[1,1,1]]
    # obstacleGrid = [[1]]

    # print(pathsWithObstacles.uniquePathsWithObstacles0(obstacleGrid))

    # grid = [[1,3,1],[1,5,1],[4,2,1]]
    # grid = [[1,2,3],[4,5,6]]
    # print(solPathSum.minPathSum(grid))

    # triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
    # triangle = [[-10]]
    #print(sol.minimumTotal(triangle))
    # print(solDP.minimumTotal(triangle))


    