# Data structure and Algorithms 

##### Binary Tree

- 前序遍历：先访问根节点-> 前序遍历左子树-> 前序遍历右子树 
- 中序遍历：先中序遍历左子树-> 根节点-> 中序遍历右子树 
- 后序遍历：先后序遍历左子树-> 后序遍历右子-> 访问根节点

递归遍历：
```golang
var res []int
func treeTraversal(root *TreeNode) []int {
    res = []int{}
    dfs(root)
    return res
} 

func dfs(root *TreeNode) {
	if root != nil {
		//res = append(res, root.Val) 前序遍历
		dfs(root.Left)
		//res = append(res, root.Val) 中序遍历
		dfs(root.Right)
		//res = append(res, root.Val) 后序遍历
	}
}
```
迭代遍历：
```golang
//前序遍历
func preorderTraversal(root *TreeNode) []int {
    if root == nil {
        return []int{}
    }
    result := make([]int, 0)
    stack := make([]*TreeNode,0)
    for len(stack) != 0 || root != nil {
        for root != nil {
            //preorder the root 先保存结果
            result = append(result, root.Val)
            stack = append(stack, root)
            root = root.Left
        }
		//pop
        node := stack[len(stack) - 1]
        stack = stack[:len(stack) - 1]
        root = node.Right
    }
    return result
}

//中序遍历
func inorderTraversal(root *TreeNode) []int {
   if root == nil {
        return []int{}
    }
    result := make([]int, 0)
    stack := make([]*TreeNode, 0)
    for len(stack) > 0 || root != nil {
        for root != nil {
            stack = append(stack, root)
            root = root.Left
        }
        node := stack[len(stack) - 1] 
        stack = stack[:len(stack) - 1]
        result = append(result, node.Val)
        root = node.Right
    }
    return result
}

//后序遍历
func postorderTraversal(root *TreeNode) []int {
    // lastVisited标识右子节点是否已弹出
    if root == nil {
        return nil
    }
    result := make([]int, 0)
    stack := make([]*TreeNode, 0)
    var lastVisited *TreeNode
    for root != nil || len(stack) != 0 {
        for root != nil {
            stack = append(stack, root)
            root = root.Left
        }
        //先不弹出
        node:= stack[len(stack)-1]
        // 根节点必须在右节点弹出之后，再弹出
        if node.Right == nil || node.Right == lastVisited {
            stack = stack[:len(stack)-1] // pop
            result = append(result, node.Val)
            // 标记节点已经弹出过
            lastVisited = node
        } else {
            root = node.Right
        }
    }
    return result
}
```

##### Linked List

删除有序链表中的重复元素 83
```golang
func deleteDuplicates(head *ListNode) *ListNode {
    current := head
    for current != nil {
        for current.Next != nil && current.Val == current.Next.Val {
            current.Next = current.Next.Next
        }
        current = current.Next
    }
    return head
}
```

删除有序链表中的重复元素 二 82
```golang
func deleteDuplicates(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}

	//判断是否head重复
	if head.Val == head.Next.Val {
		for head.Next != nil && head.Val == head.Next.Val {
			head = head.Next
		}
		return deleteDuplicates(head.Next)
	}

	head.Next = deleteDuplicates(head.Next)
	return head
}
```

反转链表 206
```golang
func reverseList(head *ListNode) *ListNode {
	// prev 是已逆转节点的head
	var prev *ListNode
	// head 是下一个被逆转的节点
	for head != nil {
		// temp保存当前head.Next, 免得head.Next被覆盖.
		temp := head.Next
		// head称为已经逆转的节点的新head
		head.Next = prev
		// prev重新变为已逆转节点的head
		prev = head
		// head指向下一个被逆转的节点
		head = temp
	}
	return prev
}
```

合并有序链表 21
```golang
//递归实现
class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if(l1 == null) {
            return l2;
        }
        if(l2 == null) {
            return l1;
        }

        if(l1.val < l2.val) {
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        } else {
            l2.next = mergeTwoLists(l1, l2.next);
            return l2;
        }
    }
}

```

判断是否是回文链表 234
```golang
//快慢指针 翻转链表
func isPalindrome(head *ListNode) bool {
    if head == nil || head.Next == nil {
        return true
    }
    slow, fast := head, head
    pre := head
    var prevpre *ListNode

    for fast != nil && fast.Next != nil {
        pre = slow
        slow = slow.Next
        fast = fast.Next.Next
        
        pre.Next = prevpre
        prevpre = pre
    }   
    if fast != nil {
        slow = slow.Next
    }
    for pre != nil && slow != nil {
        if pre.Val != slow.Val {
            return false
        }
        pre = pre.Next
        slow = slow.Next
    }
    return true
}
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



##### Stack and Queue

字符串匹配（有效的括号） 20
```golang
//左半边入栈, 右边匹配出栈否则false
func isValid(s string) bool {
    dict := map[byte]byte{')':'(', ']':'[', '}':'{'} 
    stack := make([]byte, 0)
    if s == "" {
        return true
    }

    for i := 0; i < len(s); i++ {
        if s[i] == '(' || s[i] == '[' || s[i] == '{' {
            stack = append(stack, s[i])
        } else if len(stack)>0 && stack[len(stack)-1] == dict[s[i]] {
            stack = stack[:len(stack)-1]
        } else {
            return false
        }
    }
    return len(stack) == 0
}
```



岛屿数量 200

递归DFS
```golang
func numIslands(grid [][]byte) int {
    var count int
    for i:=0;i<len(grid);i++{
        for j:=0;j<len(grid[i]);j++{
            if grid[i][j]=='1' && dfs(grid,i,j)>=1{
                count++
            }
        }
    }
    return count
}

func dfs(grid [][]byte,i,j int)int{
    if i<0||i>=len(grid)||j<0||j>=len(grid[0]){
        return 0
    }
    if grid[i][j]=='1'{
        // 标记已经访问的点
        grid[i][j]=0
        return dfs(grid,i-1,j) +
        dfs(grid,i,j-1) +
        dfs(grid,i+1,j) +
        dfs(grid,i,j+1) + 1
    }
    return 0
}
```



DFS model with stack java 

```java
boolean DFS(int root, int target) {    
  Set<Node> visited;    
  Stack<Node> s;    
  add root to s;    
  while (s is not empty) {        
    Node cur = the top element in s;        
    return true if cur is target;        
    for (Node next : the neighbors of cur) {            
      if (next is not in visited) {                
        add next to s;                
        add next to visited;            
      }        
    }        
    remove cur from s;    
  }    return false;
}
```



最大的正方形岛屿面积

字符串解码 394

```golang


```

#####  Binary Search

35 搜插位置   基础二分搜索

```golang
func searchInsert(nums []int, target int) int {
    start := 0
    end := len(nums)
    mid := 0
    
    for start < end {
        mid = start + (end-start)/2
        if target < nums[mid] {
            end = mid
        } else if target > nums[mid] {
            start = mid + 1
        } else if target == nums[mid]{
            return mid
        }
    }
    return start
}
```



#####  Dynamic Programming

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

func min(x int, y int) int {
    if x>y {
        return y
    }
    return x
}
```

65 最小路径和 经典动态规划

```golang
func minPathSum(grid [][]int) int {
	m, n := len(grid), len(grid[0])
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if i == 0 && j == 0 {
				continue
			} else if i == 0 {
				grid[i][j] = grid[i][j-1] + grid[i][j]
			} else if j == 0 {
				grid[i][j] = grid[i-1][j] + grid[i][j]
			} else {
				grid[i][j] = min(grid[i-1][j], grid[i][j-1]) + grid[i][j]
			}
		}
	}
	return grid[m-1][n-1]
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
```



[62 不同路径](https://leetcode-cn.com/problems/unique-paths/)

```golang
//经典动态规划问题 dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
//要么是从上面格子下来的，要么是从左边格子过来的
func uniquePaths(m int, n int) int {
    dp := make([][]int, m)
    for i:=0; i<m; i++ {
      	dp[i] = make([]int, n)
        for j:=0; j<n; j++ {
            //边缘上只有一条路可走
            if i==0 || j==0 {
                dp[i][j] = 1
            }else {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

            }
        }
    }
    return dp[m-1][n-1]
}
```



53 最大子序和         基础题 分治法 DP

基本DP 思想公式 `status[n+1] = max(status[n], status[n] + nums[n+1]) `

```golang
func maxSubArray(nums []int) int {
	len := len(nums)
	ret := nums[0]
	dp := nums[0]
	for i:=1; i<len; i++ {
		dp = maxV(nums[i], dp + nums[i])
		ret = maxV(ret, dp)
	}
	return ret
}

func maxV(a int, b int) int {
	if a>b {
		return a
	}
	return b
}
```

121 买卖股票

```golang
DP思想
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
PS: 只要今天比昨天贵就卖
func maxProfit(prices []int) int {
    profit := 0
    for i:=1; i< len(prices); i++ {
        if prices[i] > prices[i-1] {
            profit += prices[i] - prices[i-1]
        }
    }
    return profit
}
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



#####  Sliding window

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



#####  Backtracking
 全排列 46/47 
```golang

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



##### Others

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

