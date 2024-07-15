# Data structure and Algorithms 

用Python实现LRU 缓存

```python
from collections import OrderedDict

class LRUCache:
  def __init__(self, capacity: int):
    self.cache = OrderedDict()
    self.capacity = capacity
    
  def get(self, key: int) -> int:
    if key not in self.cache:
      return -1
    # move the key to the end
    self.cache.move_to_end(key)
    return self.cache[key]
  
  def put(self, key: int, value: int) -> None:
    if key in self.cache:
      self.cache.move_to_end(key)
    self.cache[key] = value
    if len(self.cache) > self.capacity:
      self.cache.popitem(last=False)
```



## String，Array

283 原地移动0到前面  变形（移动0到末尾）

```python
def moveZeroes(nums) -> None:
    lp, rp = len(nums)-1, len(nums)-1
    while lp >= 0 and rp >= 0:
        if nums[lp] != 0:
            nums[lp], nums[rp] = nums[rp], nums[lp]
            rp -= 1
        lp-=1
    return nums
print(moveZeroes([1, 2, 3, 0, 0]))
```

56 合并区间

```python
def merge(self, intervals: List[List[int]]) -> List[List[int]]:
    intervals.sort(key= lambda x:x[0])
    merged = []
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged
```

10.01 合并排序的数组

```python
    def merge(self, A: List[int], m: int, B: List[int], n: int) -> None:
  			# 三指针
        pa = m - 1
        pb = n - 1
        ptail = m + n - 1
        while pa>=0 or pb>=0:
            if pa == -1:
                A[ptail] = B[pb]
                pb -= 1
            elif pb == -1:
                return
            elif A[pa]<=B[pb]:
                A[ptail] = B[pb]
                pb -= 1
            else: 
                A[ptail] = A[pa]
                pa -= 1
            ptail -= 1
```

26 删除排序数组中的重复元素

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        p = 0
        for i in range(1, len(nums)):
            if nums[i] != nums[p]:
                p+=1
                nums[p] = nums[i]
        return p+1, nums

```

8/ LCR192 字符转数字

```python
    def strToInt(self, str: str) -> int:
        flag, result = 1, 0
        str = str.strip()
        for i in range(len(str)):
            if i == 0 and str[i] in ['-', '+']:
                if str[i] == '-':
                    flag *= -1
                continue
            if not str[i].isdigit():
                break 
            result = result * 10 + ord(str[i]) - ord('0')
        return min(2**31 - 1, result * flag) if flag == 1 else max(-2**31, result * flag)

```



1881 插入后的最大值

```python
  同类题目 插入5   
  def insert_five(self, a: int) -> int:
        res, sa, res = 0, str(a), None
        if a < 0:
            for i in range(1, len(sa)):
                if int(sa[i]) > 5:
                    res = sa[:i]+'5'+sa[i:]
                    return int(res)
                if i == len(sa)-1: return int(sa+'5')
      
        if a >= 0:
            for i in range(0, len(sa)):
                if int(sa[i]) < 5:
                    res = sa[:i]+'5'+sa[i:]
                    return int(res)
                if i == len(sa)-1: return int(sa+'5')
              
  
```

Can you find the triplets whose sum is zero?

```python
def findTriplets(arr, n):
    found = False
    for i in range(n - 1):
        # Find all pairs with sum
        # equals to "-arr[i]"
        s = set()
        res = []
        for j in range(i + 1, n):
            x = -(arr[i] + arr[j])
            if x in s:
                print(x, arr[i], arr[j])
								res.append([x, arr[i], arr[j]])
            else:
                s.add(arr[j])
        return res

# Driver Code
arr = [0, -1, 2, -3, 1]
n = len(arr)
findTriplets(arr, n)
```

What is the largest subset whose elements are Fibonacci numbers?

```python
def generate_fibonacci(max_value):
    fib_set = {0, 1}  # Starting with the first two Fibonacci numbers
    a, b = 0, 1
    while b <= max_value:
        a, b = b, a + b
        fib_set.add(b)
    return fib_set

def largest_fibonacci_subset(input_set):
    max_value = max(input_set)
    fib_set = generate_fibonacci(max_value)
    return input_set.intersection(fib_set)

# Example usage:
input_numbers = {0, 1, 2, 3, 4, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377}
largest_subset = largest_fibonacci_subset(input_numbers)
print(largest_subset)
```

Calculate the maximum value using the '+' or '*' sign between two numbers in the given string.

```python
def calcMaxValue(str): 
  
    # Store first character as integer 
    # in result 
    res = int(str[0])
  
    # Start traversing the string  
    for i in range(1, len(str)): 
          
        # Check if any of the two numbers  
        # is 0 or 1, If yes then add current  
        # element 
        if(str[i] == '0' or
           str[i] == '1' or res < 2): 
            res += int(str[i])
        else: 
            res *= int(str[i]) 
    return res  
```



## Binary Tree

- 前序遍历：先访问根节点-> 前序遍历左子树-> 前序遍历右子树 
- 中序遍历：先中序遍历左子树-> 根节点-> 中序遍历右子树 
- 后序遍历：先后序遍历左子树-> 后序遍历右子-> 访问根节点

递归遍历：
```python
    #python
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        def inorder(root):
            if not root:
                return
            res.append(root.val) #前序
            inorder(root.left)
            res.append(root.val) #中序
            inorder(root.right)
            res.append(root.val) #后序
        inorder(root)
        return res
```


102 二叉树的层序遍历






236 二叉树的最近公共祖先 

```python3
def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:

	# 如果 p和q中有等于 root的，那么它们的最近公共祖先即为root（一个节点也可以是它自己的祖先）
	if not root or root == p or root == q: return root
	# 递归遍历左右子树，只要在子树中找到了p或q，则先找到谁就返回谁
	left = self.lowestCommonAncestor(root.left, p, q)
	right = self.lowestCommonAncestor(root.right, p, q)

	# 当 left和 right均不为空时，说明 p、q节点分别在 root异侧, 最近公共祖先即为 root
	if not left and not right: return 
	# 如果在左子树中 p和 q都找不到，则 p和 q一定都在右子树中，右子树中先遍历到的那个就是最近公共祖先（一个节点也可以是它自己的祖先
	if not left: return right
	# 如果 left不为空，在左子树中有找到节点（p或q），这时候要再判断一下右子树中的情况，如果在右子树中，p和q都找不到，则 p和q一定都在左子树中，左子树中先遍历到的那个就是最近公共祖先（一个节点也可以是它自己的祖先）
	if not right: return left

	return root 
```



迭代遍历：



找到二叉树的最小公共祖先



## Linked List

83 删除有序链表中的重复元素

```python
def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        cur = head
        while cur:
            while cur.next and cur.val == cur.next.val:
                cur.next =  cur.next.next
            cur = cur.next
        return head

```

删除有序链表中的重复元素 二 82
```python
def deleteDuplicates(head):
    pseudo = prev = ListNode(None)
    pseudo.next = head
    node = head
    while node:
        if node.next and node.val == node.next.val:
            dupl_value = node.val
            node = node.next
            while node and node.val == dupl_value:
                node = node.next
            prev.next = None
        else: 
            prev.next = node
            prev = node
            node = node.next
    return pseudo.next
```

反转链表 206

```python
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # prev 是所有已经逆转的节点的head
        prev = None
        while head is not None:
            tmp = head.next
            head.next = prev
            prev = head
            head = tmp
        return prev
```

反转链表by every 2 24

```golang
func swapPairs(head *ListNode) *ListNode {
    if head == nil {
        return nil
    }
    //头节点增加dummy head
  	dummy := &ListNode{}
		dummy.Next = head
    prev := dummy
    //单双数终止检查
		for head != nil && head.Next != nil {
    	//prev->a->b->c
			b := head.Next //暂存b
			head.Next = b.Next //连接a->c
			b.Next = head //翻转b->a
      //将prev指向翻转后的当前头节点b
			prev.Next = b

      //因为两两交换 b->a(head)->c->d
      //prev 变为 a
			prev = head
      //推进head指针到c,开始下一个翻转循环
			head = head.Next
		}
		return dummy.Next
}
```


反转链表by every K 25 - 与上题思路类似，需注意边界条件和断链重连

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        dummy = ListNode(0)
        dummy.next = head
        pre, end = dummy, dummy
        
        while end.next:
            for i in range(k):
                if end:
                    end = end.next
            if not end:
                break
            #break the chain
            start = pre.next
            tmp = end.next
            end.next = None
            
            #reverse it and fit back
            pre.next = self.reverseLL(start)
            start.next = tmp
            
            # enter next k loop
            pre = start
            end = pre
            
        return dummy.next
```



合并有序链表 21

```python
//递归实现
class Solution(object)
    def mergeTwoLists(self, l1, l2):
        prev = dummy = ListNode(None)

       	while l1 and l2:
            if l1.val < l2.val:
                prev.next = l1
                l1 = l1.next
            else:
                prev.next = l2
                l2 = l2.next
            prev = prev.next
        prev.next = l1 or l2 # link prev to the list with remaining nodes
        return dummy.next

```

判断是否是回文链表 234
```python
//快慢指针 翻转链表
def isPalindrome(head):
    fast, slow = head, head
    rev = None
    
    while fast and fast.next:
        fast = fast.next.next
        next_slow = slow.next
        slow.next = rev
        rev = slow
        slow = next_slow
    if fast:
    		slow = slow.next
    while slow:
    		if slow.val != rev.val:
          	return False
        slow = slow.next
        rev = rev.next
    return True
```

判断链表中是否有环 141
```golang
//快慢指针
func hasCycle(head *ListNode) bool {
    if head == nil || head.Next == nil || head.Next.Next == nil{
        return false
    }
    slow, fast := head, head.Next

    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
        if fast == slow {
            return true
        }
    }
    return false
}
```

160 相交链表 -- 求长度差

```golang
func getIntersectionNode(headA, headB *ListNode) *ListNode {
    if headA==nil || headB==nil {
        return nil
    }
    tmpNode := headA ;
    lengthA := 0
    for true {
        if tmpNode.Next != nil {
            tmpNode = tmpNode.Next
            lengthA++
        } else {
            break
        }
    }
    tmpNode = headB
    lengthB := 0
    for true {
        if tmpNode.Next != nil {
            tmpNode = tmpNode.Next
            lengthB++
        } else {
            break
        }
    }

    if lengthA >= lengthB {
            for i := 0; i < lengthA - lengthB; i++ {
                headA = headA.Next;
            }
        } else {
            for i := 0; i < lengthB - lengthA; i++ {
                headB = headB.Next;
            }
        }

    for headA != nil {
        if headA == headB {
            return headA
        } 
        headA = headA.Next
        headB = headB.Next
    }
    return nil
}
```



## Stack and Queue

字符串匹配（有效的括号） 20
```python
//左半边入栈, 右边匹配出栈否则false
def isValid(s):
  	dic = {'(': ')', '[': ']', '{': '}'}
  	stack = []
  
  	for char in s:
      	if char in dic:
          	stack.append(char)
        else:
          	if not stack or dic[stack.pop()] != char:
              	return False
    return not stack
```



## Two Pointers 双指针

80 删除有序数组中的重复项 2
```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        l = 1
        for r in range(2, len(nums)):
            #和nums[left]比, 还要和nums[left - 1]比，保证元素可以重复两次
            if nums[r] == nums[l] and nums[r] == nums[l-1]:
                continue
            l += 1
            nums[l] = nums[r] 
        return l + 1
```

986 区间列表的交集

```golang
func intervalIntersection(firstList [][]int, secondList [][]int) [][]int {
    //corner case
    res := [][]int{}
    if len(firstList) == 0 || len(secondList) == 0 {
        return res
    }
    
    idx1, idx2 := 0, 0
    for idx1 < len(firstList) && idx2 < len(secondList) {
            start := compare(firstList[idx1][0], secondList[idx2][0], true)
            end := compare(firstList[idx1][1], secondList[idx2][1], false)
            if start <= end {
                res = append(res, []int{start, end})
            }
            //谁先结束, 谁的指针步进，考虑多重合区间的问题
            if firstList[idx1][1] < secondList[idx2][1] {
                idx1 += 1
            } else {
                idx2 += 1
            }
    }
    return res

}

func compare(x int, y int, max bool) int {
    if max == true {
        if x > y {
            return x
        }
        return y
    } else if max == false {
        if x < y {
            return x
        }
        return y
    }
    return y
}
```

11 盛水最多的容器

```python
def maxArea(height):
  	left = 0
  	right = len(height)-1
    max_area = (right - left) * min(height[right], height[left])
    while left < right:
    		if height[left] < height[right]:
            left += 1
        else:
          	right -= 1
        max_area = max(max_area, (right - left) * min(height[right], height[left]))
    return max_area
```

415 字符串相加

```python
		def addStrings(num1: str, num2: str) -> str:
				#双指针模拟
        res = []
        add = 0 #存储是否进位
        x, y = len(num1)-1, len(num2)-1
        while(x >= 0 or y >= 0):
          	#0补位如果长度不同
            av = 0 if x<0 else int(num1[x])
            bv = 0 if y<0 else int(num2[y])
            sums = av + bv + add
            #其他进制把10改相应即可
            res.append(str(sums%10))
            add = 1 if sums>=10 else 0
            #头部进位
            if x <= 0 and y <= 0 and add == 1:
                res.append(str(add))
            x -= 1
            y -= 1
        return ''.join(reversed(res))		
```



## DFS

岛屿数量 200 延伸问题 695最大岛屿面积 463 岛屿周长

递归DFS
```python
def numIslands(grid):
    if not grid:
        return 0
    count = 0
    
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1' and scan(grid, i, j)>=1:
                count += 1 
		return count
  
def scan(grid, i, j):
    if i<0 or i >= len(grid) or j<0 or j >= len(grid[0]):
        return 0
      
    if grid[i][j] == '1':
        grid[i][j] = 0
        return scan(grid, i-1, j) + scan(grid, i, j-1) + scan(grid, i+1, j) + scan(grid, i, j+1) + 1
    return 0
```

```python
Target SUM
DFS
class Solution:
    def findTargetSumWays(self, nums: List[int], S: int) -> int:
        d = {}
        def dfs(cur, i, d):
            if i < len(nums) and (cur, i) not in d: # 搜索周围节点
                d[(cur, i)] = dfs(cur + nums[i], i + 1, d) + dfs(cur - nums[i],i + 1, d)
            return d.get((cur, i), int(cur == S))   
        return dfs(0, 0, d)


class Solution:
    def findTargetSumWays(self, nums: List[int], S: int) -> int:
        if sum(nums) < S or (sum(nums) + S) % 2 == 1: return 0
        P = (sum(nums) + S) // 2
        dp = [1] + [0 for _ in range(P)]
        for num in nums:
            for j in range(P,num-1,-1):dp[j] += dp[j - num]
        return dp[P]
```



463 岛屿周长

```python
class Solution:
    def islandPerimeter(self, grid: List[List[int]]) -> int:
        rowlen = len(grid)
        if not grid or rowlen == 0:
            return 0
        collen = len(grid[0])
        res = 0
        for i in range(rowlen):
            for j in range(collen):
                if grid[i][j] == 1:
                    res += 4
                    if i-1>=0 and grid[i-1][j] == 1:
                        res -= 2
                    if j-1>=0 and grid[i][j-1] == 1:
                        res -= 2
        return res
```



##  Binary Search

35 搜插位置   基础二分搜索

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        l, m, r = 0, 0, len(nums)
        while l<r:
            m = int(l + (r-l)/2)
            if nums[m] > target:
                r = m
            elif nums[m] < target:
                l = m+1
            elif nums[m] == target:
                return m
        return l
```



74 搜索二维矩阵

```golang
func searchMatrix(matrix [][]int, target int) bool {
    //corner case 
    if len(matrix) == 0 || matrix == nil {
        return false
    }

    arr := make([]int, 0)
    for _, v := range matrix {
        arr = append(arr, v...)
    }

    return bSearch(arr, 0, len(arr)-1, target)
}

//二分搜索基本模板
func bSearch(nums []int, start int, end int, target int) bool {
    for start < end {
        mid := start + (end-start)/2
        if nums[mid] >= target {
            end = mid
        } else {
            start = mid + 1
        }
    }

    if nums[end] == target {
        return true
    } else {
        return false
    }
}
```

33 搜索旋转排序数组

```golang
func search(nums []int, target int) int {
    lens := len(nums)
    //corner case
    if lens == 1 {
        if target == nums[0] {
            return 0
        } 
        return -1
    }
    //find twist point
    tPoint := 0
    for i:=1;i<lens;i++ {
        if nums[i-1] > nums[i] {
            tPoint = i-1
        }
    }

    if res := binarySearch(nums, 0, tPoint, target); res != -1 {
        return res
    } else {
        return binarySearch(nums, tPoint+1, lens-1, target)
    }
}

func binarySearch(nums []int, left int, right int, target int) int {
    mid := 0
    for left < right {
        mid = left + (right-left)/2
        if nums[mid] >= target {
            right = mid
        } else {
            left = mid + 1
        }
    }
    if nums[right] == target {
        return right
    } else {
        return -1
    }
}
```


##  Dynamic Programming 动态规划

面试题 08.11 分硬币

```python
class Solution:
    def waysToChange(self, n: int) -> int:
        # 动态规划, dp[sm]存总和为sm的方案数
        # 对于每一种硬币c, 都有dp[sm]=dp[sm]+dp[sm-c] (c<=sm<=n)
        MOD = 1000000007
        coins = [1, 5, 10, 25]
        dp = [1] + [0] * n
        for c in coins:
            for sm in range(c, n + 1):
                dp[sm] = (dp[sm] + dp[sm - c]) % MOD
        return dp[n]
```

264 丑数2  --  三指针 + DP

```golang
func nthUglyNumber(n int) int {
    dp := make([]int, n+1)
    dp[1] = 1
    x, y, z := 1, 1, 1
    for i:=2; i<n+1; i++ {
        x2, x3, x5 := dp[x]*2, dp[y]*3, dp[z]*5
        dp[i] = min(min(x2, x3), x5)
        if dp[i] == x2 {
            x++
        } 
        if dp[i] == x3 {
            y++
        }
        if dp[i] == x5 {
            z++
        }
    }
    return dp[n]
}
```

64 最小路径和 经典动态规划

```python
#python
class Solution:
    def minPathSum(self, grid: [[int]]) -> int:
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if i == j == 0: 
                    continue
                #只能从左边过来
                elif i == 0:  
                    grid[i][j] = grid[i][j - 1] + grid[i][j]
                #只从上边下来
                elif j == 0:  
                    grid[i][j] = grid[i - 1][j] + grid[i][j]
                else: 
                    #都可能，取小的
                    grid[i][j] = min(grid[i - 1][j]+ grid[i][j], grid[i][j - 1]+ grid[i][j]) 
        return grid[-1][-1]
```



[62 不同路径](https://leetcode-cn.com/problems/unique-paths/)

```golang
//经典动态规划问题 dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
//要么是从上面格子下来的，要么是从左边格子过来的
```

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [ s for s in range(n) ]
        for i in range(m):
            for j in range(n):
                if i == 0 or j == 0:
                    dp[j] = 1
                else:
                    dp[j] = dp[j] + dp[j-1]
        return dp[n-1]
```


53 最大子序和 最大子数组和        基础题 分治法 DP

基本DP 思想公式 `status[n+1] = max(status[n], status[n] + nums[n+1]) `

```python
python 空间优化后
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        #dp 记录, 不用记录所有数值，只用记录最大值
        dp = nums[0]
        maxSum = nums[0]
        for i in range(1, len(nums)):
            dp = max(nums[i], dp + nums[i])
            if dp > maxSum:
                maxSum = dp
        
        return maxSum
```

97 交错字符串
```python
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        #二维动态规划
        len1 = len(s1)
        len2 = len(s2)
        len3 = len(s3)
        if len1 + len2 != len3:
            return False
        dp = [ [False]*(len2+1) for i in range(len1+1) ]
        dp[0][0] = True
        for i in range(1, len1+1):
            dp[i][0] = (dp[i-1][0] and s1[i-1] == s3[i-1])
        for i in range(1, len2+1):
            dp[0][i] = (dp[0][i-1] and s2[i-1] == s3[i-1])
        for i in range(1, len1+1):
            for j in range(1,len2+1):
                dp[i][j]= (dp[i][j-1] and s2[j-1] == s3[i+j-1]) or (dp[i-1][j] and s1[i-1]==s3[i+j-1])
        return dp[-1][-1]

```

121 买卖股票

```python
def max_profit_with_days(prices):
    if not prices:
        return 0, None, None

    min_price = prices[0]
    max_profit = 0
    buy_day = 0
    sell_day = 0

    for i, price in enumerate(prices):
        if price < min_price:
            min_price = price
            buy_day = i
        current_profit = price - min_price
        if current_profit > max_profit:
            max_profit = current_profit
            sell_day = i

    return max_profit, prices[buy_day], prices[sell_day]
```

121 买卖股票

```golang
//DP思想
func maxProfit(prices []int) int {
    profit := 0
	buyPrice := prices[0]
	for i :=1 ; i< len(prices); i++ {

		if p := prices[i] - buyPrice; p > profit {
			profit = p
		}
		if prices[i] < buyPrice {
			buyPrice = prices[i]
		}
    }
    return profit
}
```

122 买卖股票2 -- 贪心算法

```golang
// 只要今天比昨天贵就卖
func maxProfit(prices []int) int {
    profit := 0
    for i:=1; i< len(prices); i++ {
        if prices[i] > prices[i-1] {
            profit += prices[i] - prices[i-1]
        }
    }
    return profit
}

def maxProfit(self, prices: List[int]) -> int:
        #从第二天开始，如果当天股价大于前一天股价，则在前一天买入，当天卖出，即可获得利润。如果当天股价小于前一天股价，则不买入，不卖出。也即是说，所有上涨交易日都做买卖，所有下跌交易日都不做买卖，最终获得的利润最大

        profit = 0
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                profit += prices[i] - prices[i-1]
        return profit
```

```

```



45 跳跃游戏 2 -贪心算法

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        #正向边界贪心跳越
        maxP, end, st = 0, 0, 0
        for i in range(len(nums) -1):
            if maxP >= i:
                maxP = max(maxP, i + nums[i])
                if i == end:
                    end = maxP
                    st += 1
        return st

```

322 零钱兑换 - 背包问题

```golang
//类似背包问题，可使用动态规划解决
//转移方程： f(n) = min(f(n - c1), f(n - c2), ... f(n - cn)) + 1
func coinChange(coins []int, amount int) int {
    if coins == nil || len(coins) == 0 {
        return -1
    }
    res := make([]int, amount+1)
    for i:=1; i < amount+1; i++ {
        res[i] = math.MaxInt32
        for _, v := range coins {
            if i - v >= 0 {
                res[i] = min(res[i], res[i-v]+1)   
            }
        }
    }

    if res[amount] == math.MaxInt32 {
        return -1
    }
    return res[amount]
}
```

AcWing 487 金明的预算方案

```python
v, w = [], []
for i in range(n):
  x=[int(j) for j in input().split()]
  v.append(x[0])
  w.append(x[1])
  
def max_buy(w, v):
  # 购买数量，总钱数
  n, m = w[0], v[0] 
  f = [[0] * (m+1) for _ in range(n+1)]
  for i in range(1, n+1):
    val = v[i]*w[i]
    for j in range(1, m+1):
      f[i][j] = f[i-1][j]
      if j >= v[i]:
        f[i][j] = max(f[i][j], f[i-1][j-v[i]]+val)
	return f[n][m]
```



##  Sliding window

239 [滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum/)

维护一个队列存储最大值

```golang
func maxSlidingWindow(nums []int, k int) []int {
    if len(nums) == 0 {
        return []int{}
    }

    //维护一个有序切片
    window := make([]int, 0)
    res := make([]int, 0)
    for i:=0; i<len(nums); i++ {
        //循环删除维护队列尾部最小元素如果其小于当前元素
        for i>0 && len(window)>0 && nums[i] > window[len(window)-1] {
            window = window[:len(window)-1]
        }
      
        window = append(window, nums[i])
        //如果窗口尾端值为最大值，推出维护队列
        if i >= k && nums[i-k] == window[0] {
            window = window[1:]
        }
           
        //将当前最大值写入结果
        if i >= k-1 {
            res = append(res, window[0])
        } 
    }
    return res
}
```

3 无重复字符的最长子串 

```golang
func lengthOfLongestSubstring(s string) int {
    byt := []byte(s)
    if len(byt) == 0 {
        return 0
    }
    hmap := make(map[byte]int) 
    res := 0
    start := 0
    for i:=0; i<len(byt); i++ {
        if _, ok := hmap[byt[i]]; ok {
            start = max(start, hmap[byt[i]] + 1) //有重复字符推进维护窗口
        }
        hmap[byt[i]] = i  //更新map 存index，key为字符
        res = max(res, i - start + 1) 
    }
    return res
}

func max(x, y int) int {
    if x > y {
        return x
    }
    return y
}
```

713 乘积小于K子数组

```golang
func numSubarrayProductLessThanK(nums []int, k int) int {
    //corner case
    if len(nums)==0 || k == 0 || k==1 {
        return 0
    }
    //滑动窗口双指针
    l, product, res := 0, 1, 0
    for r:=0; r<len(nums); r++ {
        product *= nums[r]
        for product >= k {
            product /= nums[l]
            l += 1
        }
        res += r-l+1
    }
    return res
}
```

209 长度最小子数组

```golang
func minSubArrayLen(target int, nums []int) int {
    //corner case
    if len(nums) == 0 || nums == nil {
        return 0
    }
    //滑动窗口
    //golang最大数表达 int(^uint(0) >> 1)
    res, sum, l, length := int(^uint(0) >> 1), 0, 0, 0
    
    for r:=0; r<len(nums); r++ {
        sum += nums[r]
        for sum >= target {
            //length
            length = r-l+1
            if res > length {
                res = length
            }
            sum -= nums[l] //不断调整起始点位置
            l += 1
        }
    }
    if res == int(^uint(0) >> 1) {
        return 0
    } else {
        return res
    }
}
```



##  Backtracking
 全排列 46/47 
```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []  
        path = []
        def backtrack(nums):
            if len(path) == len(nums):
                return res.append(path[:])  #找到了一组
            for i in range(0,len(nums)):
                if nums[i] in path:  #path已经收录的元素，跳过
                    continue
                path.append(nums[i])
                backtrack(nums)  #递归
                path.pop()  #回溯
        backtrack(nums)
        return res
```



22 括号生成

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]: 
        ans = []
        path = []
        def backtrack(left,right):
            '''
            left = 0 # 左括号数
            right = 0 # 右括号数
            '''
            if len(path) == 2*n :
                ans.append("".join(path))
                return
            # 右括号是否可选为： left-right > 0 ? 可选右:不可选
            if left - right > 0:
                path.append(")")
                backtrack(left, right+1)
                path.pop()

            # 左括号是否可选为： n - left > 0? 可选左：不可选
            if n - left > 0:
                path.append("(")
                backtrack(left+1, right)
                path.pop()
        backtrack(0,0)
        return ans
```



78 子集

```golang
//遍历，遇到一个数就把所有子集加上该数组成新的子集
func subsets(nums []int) [][]int {
    res := make([][]int, 1, int(math.Pow(2, float64(len(nums)))) + 1)
    res[0] = []int{}
    for _, ar := range nums {
        for _, v := range res {
            newV := make([]int, len(v), len(v)+1)
            //不能直接append编辑res,因为会改变res所指向的内存地址
            //深拷贝一个newV再append
            copy(newV, v)
            res = append(res, append(newV, ar))
        }
    }
    return res
}
```



90 子集2 - 回溯

```golang
func subsetsWithDup(nums []int) (res [][]int) {
	var dfs func(temp []int, idx int)
	n := len(nums)
	sort.Ints(nums)
	dfs = func(temp []int, idx int) {
		res = append(res, append([]int(nil), temp...))
		for i := idx; i < n; i++ {
			if i > idx && nums[i] == nums[i-1] {
				continue
			}
			temp = append(temp, nums[i])
			dfs(temp, i+1)
			temp = temp[:len(temp)-1]
		}
	}
	dfs([]int{}, 0)
	return
}
```



## Others

位运算 

136 2N+1 找1

```java
class Solution {
    public int singleNumber(int[] nums) {
        int res = 0;
        for(int i=0; i < nums.length; i++) {
            res = res^nums[i];
        }
        return res;
    }
}
```

2N 找 2个不重复的

解1.哈希数组转存，值当下标，次数当value，最后遍历出value为1的结果。

解2. 全部异或后，根据结果位分两组再异或，结果就是两个不重复的数。

```java
    public int[] singleNumber(int[] nums) {
        int [] res = new int[2];
        int tmp = 0;
        for (int i = 0; i < nums.length; i++) {
            tmp ^= nums[i];
        }
        
        int temp = lowbit(tmp);
        for (int i = 0; i < nums.length; i++) {
            if((temp & nums[i]) == 0) res[0] ^= nums[i];
            else res[1] ^= nums[i];
        }
        return res;
        
    }
    public int lowbit(int x){
        return x & -x;
    }
```

212 找重复 List Set HashTable

```Golang
func containsDuplicate(nums []int) bool {
    set := map[int]struct{}{}  //用map模拟set, 赋予空结构体
    for _, v := range nums {
        if _, has := set[v]; has {
            return true
        }
        set[v] = struct{}{}
    }
    return false
}
```

349 求两个无序数组交集 - >哈希表     延伸：两个有序数组交集->双指针一次遍历

```
func intersection(nums1 []int, nums2 []int) []int {
    mp := make(map[int]int)
    res := make([]int, 0)
    for _, v := range nums1 {
        if mp[v] == 0 {
            mp[v]++
        } 
    }
    for _, v := range nums2 {
        if mp[v] == 1 {
            res = append(res, v)
            mp[v]--
        }
    }
    return res
}
```

5 最长回文子串 中心扩散法  647题类似

```golang
func longestPalindrome(s string) string {
    if s == "" || len(s) == 0 {
            return "";
        }
    start, end := 0, 0
    maxS := 0
    maxLen := 0
    length := 1
    for k:=0; k<len(s); k++ {
        start, end = k-1, k+1
        //向左扩散
        for start >= 0 && s[start] == s[k] {
            start--
            length++
        }
        //向右扩散
        for end < len(s) && s[end] == s[k] {
            end++
            length++
        }
        //向两边同时扩散
        for start >= 0 && end < len(s) && s[start] == s[end] {
            start--
            end++
            length += 2
        }
        //更新最大回文len和indexs
        if length > maxLen {
            maxLen = length
            maxS = start
        }
        //重置长度
        length = 1
    }
    return s[maxS+1 : maxS+maxLen+1]
}
```
```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        if len(s) == 1: return s
        max_len = 0
        leng = 1
        max_s = 0
        for k in range(len(s)):
            l = k-1
            r = k+1
            while l>=0 and s[l]==s[k]:
                l-=1
                leng+=1

            while r<len(s) and s[r]==s[k]:
                r+=1
                leng+=1
            
            while l>=0 and r< len(s) and s[l]==s[r]:
                l-=1
                r+=1
                leng+=2
                
            if leng > max_len:
                max_len = leng
                max_s = l
            leng = 1
        return s[max_s+1:max_s+max_len+1]
```

实现 LRU 缓存
Python

