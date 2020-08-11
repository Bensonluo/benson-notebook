# Benson-notebook

### Table of Contents

- [Benson-notebook](#benson-notebook)
    - [Table of Contents](#table-of-contents)
    - [Data structure and Algorithm](#data-structure-and-algorithm)
        - [Binary Tree](#binary-tree)
        - [Linked List](#linked-list)
        - [Stack and Queue](#stack-and-queue)
        - [Binary representation](#binary-representation)
        - [Binary Search](#binary-search)
        - [Sorting algorithm](#sorting-algorithm)
        - [Dynamic Programming](#dynamic-programming)
        - [Sliding window](#sliding-window)
        - [Backtracking](#backtracking)
    - [Go programing language](#go-programing-language)
        - [Basic](#basic)
        - [Slice](#slice)
        - [Hashmap](#hashmap)
        - [Struct](#struct)
        - [Marshaling](#marshaling)
        - [Garbage collection](#garbage-collection)
        - [Exception handing](#exception-handing)
        - [Go routine](#go-routine)
        - [Go scheduler](#go-scheduler)
    - [Redis](#redis)
    - [MIT 6.824 Distributed Systems Spring 2020](#mit-6824-distributed-systems-spring-2020)
    - [Interview Questions](#interview-questions)
    - [Reading List](#reading-list)
    - [Support or Contact](#support-or-contact)
  
----------

### Data structure and Algorithm

[My Leetcode account](https://leetcode-cn.com/u/peng-194/)

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

DFS model with stack

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
    }
    return false;
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

字符串解码 394
```golang


```



#####  Binary representation

#####  Binary Search

#####  Sorting algorithm

#####  Dynamic Programming

#####  Sliding window

#####  Backtracking
 全排列 46/47 
```golang

```


----------


### Go programing language
##### Basic

- Variable declaration
``` golang
s := ""
var s string
var s = ""
var s string = ""
```

第一种形式，是一条短变量声明，最简洁，但只能用在函数内部，而不能用于包变量。
第二种形式依赖于字符串的默认初始化零值机制，被初始化为""。
第三种形式用得很少，除非同时声明多个变量。
第四种形式显式地标明变量的类型，当变量类型与初值类型相同时，类型冗余，但如果两者类型不同，变量类型就必须了。实践中一般使用前两种形式中的某个，初始值重要的话就显式地指定变量的类型，否则使用隐式初始化。

- For loop

```golang
for key, value := range oldMap {
    newMap[key] = value
}
```

- Binary operator

```golang
//Go语言中算术运算、逻辑运算和比较运算的二元运算符
//它们按照先级递减的顺序的排列：
* / % << >> & &^
+ - | ^
== != < <= > >=
&&
||
```

- Pointer

```golang
x := 1 
p := &x // p, of type *int, points to x (指针)
fmt.Println(*p) // "1"
*p = 2 // equivalent to x = 2 (指向变量内存地址的值)
fmt.Println(x) // "2"
```
- Deferred

在调用普通函数或方法前加上关键字defer，就完成了defer所需要的语法， defer后面的函数会被延迟执行，且不论包含defer语句的函数是正常结束，还是异常结束。

一个函数中执行多条defer语句，它们的执行顺序与声明顺序相反。

defer语句经常被用于处理成对的操作，如**打开、关闭、连接、断开连接、加锁、释放锁**。通过defer机制，不论函数逻辑多复杂，都能保证在任何执行路径下，资源被释放。释放资源的defer应该直接跟在请求资源的语句后。

*还可用于打开关闭文件，操作互斥锁，调试复杂程序是用于记录进入和退出函数的时间。*

##### Slice

- slice 不可比较

- appendInt 函数

``` golang
func appendInt(x []int, y int) []int {
	var z []int
	zlen := len(x) + 1
	if zlen <= cap(x) {
		// There is room to grow. Extend the slice.
		z = x[:zlen]
	} else {
		// There is insufficient space. 
		// Allocate a new array.
		// Grow by doubling.
		zcap := zlen
		if zcap < 2*len(x) {
			zcap = 2 * len(x)
	}
	z = make([]int, zlen, zcap)
	copy(z, x) // a built-in function;
	}
	z[len(x)] = y
	return z
}
```
通常我们并不知道append调用是否导致了内存的重新分配，因此，通常是将append返回的结果直接赋值给输入的slice变量：runes = append(runes, r)

更新slice变量不仅对调用append函数是必要的，实际上对应任何可能导致长度、容量或底层数组变化的操作都是必要的。要正确地使用slice，需要记住尽管底层数组的元素是间接访问的，但是slice对应结构体本身的指针、长度和容量部分是直接访问的。要更新这些信息需要像上面例子那样一个显式的赋值
操作。从这个角度看，slice并不是一个纯粹的引用类型，它实际上是一个聚合类型。

- slice可以用来模拟stack
  
```golang
//使用append函数将新的值压入stack：
stack = append(stack, v) // push v
//stack的顶部位置对应slice的最后一个元素：
top := stack[len(stack)-1] // top of stack
//通过收缩stack可以弹出栈顶的元素
stack = stack[:len(stack)-1] // pop
//要删除slice中间的某个元素并保存原有的元素顺序
//通过内置的copy函数将后面的子slice向前依次移动一位完成：
func remove(slice []int, i int) []int {
	copy(slice[i:], slice[i+1:])
	return slice[:len(slice)-1]
}
func main() {
	s := []int{5, 6, 7, 8, 9}
	fmt.Println(remove(s, 2)) // "[5 6 8 9]"
}
//用最后一个元素覆盖被删除的元素：
func remove(slice []int, i int) []int {
	slice[i] = slice[len(slice)-1]
	return slice[:len(slice)-1]
}
func main() {
	s := []int{5, 6, 7, 8, 9}
	fmt.Println(remove(s, 2)) // "[5 6 9 8]
}
```

- 模拟队列
  
``` golang
// 创建队列
queue:=make([]int,0)
// enqueue入队
queue=append(queue,10)
// dequeue出队
v:=queue[0]
queue=queue[1:]
// 长度0为空
len(queue)==0
```

##### Hashmap 
``` golang
// 创建
map:=make(map[string]int)
// 设置kv
map["key"]=1
// 删除k 失败返0
delete(map,"key")
// 遍历
for k,v:=range map{
    println(k,v)
}
```
- map 元素不能取址操作，原因是map可能随着元素数量的增长而重新分配更大的内存空间，从而可能导致之前的地址无效
- map 键需要可比较，不能为 slice、map、function
- map 值都有默认值，可以直接操作默认值，如：m[age]++ 值由 0 变为 1
- 比较两个 map 需要遍历，其中的 kv 是否相同，因为有默认值关系，所以需要检查 val 和 ok 两个值

##### Struct
- 通过点操作符访问, 或者是对成员取地址，然后通过指针访问
- 结构体成员的输入顺序也有重要的意义
- 大写字母开头的，那么该成员就是导出
- 一个命名为S的结构体类型将不能再包含S类型的成员（该限制同样适应于数组。）但是S类型的结构体可以包含*S指针类型的成员
- 创建并初始化一个结构体变量，并返回结构体的地址：```pp := &Point{1, 2}```
- 结构体的全部成员都是可以比较的，那么结构体也是可以比较的，== 会比较结构体的每一个成员
  
##### Marshaling
``` golang
data, err := json.Marshal(movies)
data, err := json.MarshalIndent(movies, "", " ") //带缩进
if err != nil {
	log.Fatalf("JSON marshaling failed: %s", err)
}
fmt.Printf("%s\n", data)
```

##### Garbage collection
Go语言的自动垃圾收集器从每个包级的变量和每个当前运行函数的每一个局部变量开始，通过指针或引用的访问路径遍历，是否可以找到该变量。如果不可达 -> 回收

注意：如果将指向短生命周期对象的指针保存到具有长生命周期的对象中，特别是保存到全局变量时，会阻止对短生命周期对象的垃圾回收（从而可能影响程序的性能）。

##### Exception handing
 - Go使用控制流机制（如if和return）处理异常
 - 错误处理策略: 向上传播/重试/输出并结束/输出不中断/忽略

##### Go routine

Go语言通过goroutine提供了对于并发编程的最清晰最直接的支持，Go routine 特性小结：

1. goroutine是Go语言运行库的功能，不是操作系统提供的功能，go routine不是用线程实现的。具体可参见Go语言源码里的pkg/runtime/proc.c

2. go routine就是一段代码，一个函数入口，以及在堆上为其分配的一个堆栈。所以它非常廉价，我们可以很轻松的创建上万个goroutine，但它们并不是被操作系统所调度执行

3. 除了被系统调用阻塞的线程外，Go运行库最多会启动$GOMAXPROCS个线程来运行goroutine

4. go routine是协作式调度的，如果go routine会执行很长时间，而且不是通过等待读取或写入channel的数据来同步的话，就需要主动调用Go sched()来让出CPU

5. 和所有其他并发框架里的协程一样，go routine里所谓“无锁”的优点只在单线程下有效，如果$GOMAXPROCS > 1并且协程间需要通信，Go运行库会负责加锁保护数据，这也是为什么sieve.go这样的例子在多CPU多线程时反而更慢的原因

6. Web等服务端程序要处理的请求从本质上来讲是并行处理的问题，每个请求基本独立，互不依赖，几乎没有数据交互，这不是一个并发编程的模型，而并发编程框架只是解决了其语义表述的复杂性，并不是从根本上提高处理的效率，也许是并发连接和并发编程的英文都是concurrent吧，很容易产生“并发编程框架和coroutine可以高效处理大量并发连接”的误解。

7. Go语言运行库封装了异步IO，所以可以写出貌似并发数很多的服务端，可即使我们通过调整$GOMAXPROCS来充分利用多核CPU并行处理，其效率也不如我们利用IO事件驱动设计的、按照事务类型划分好合适比例的线程池。在响应时间上，协作式调度是硬伤。

8. Go routine最大的价值是其实现了并发协程和实际并行执行的线程的映射以及动态扩展，随着其运行库的不断发展和完善，其性能一定会越来越好，尤其是在CPU核数越来越多的未来，终有一天我们会为了代码的简洁和可维护性而放弃那一点点性能的差别。

- **Channels**

```golang
ch := make(chan int) // ch has type 'chan int'
ch = make(chan int)    // unbuffered channel
ch = make(chan int, 0) // unbuffered channel
ch = make(chan int, 3) // buffered channel with capacity 3

ch <- x  // a send statement
x = <-ch // a receive expression in an assignment statement
<-ch     // a receive statement; result is discarded

close(ch) // close a channel, panic if still sending
```
   
  1. channels 是Goroutine 之间传递消息的通信机制， channels都有一个特殊的类型，也就是channels可发送数据的类型。
  2. 创建channel ```ch := make(chan int) ```。
  3. 和map 一样， channels也对应一个make创建的底层数据结构的引用。
  4. channels 可以用==来比较，如果引用相同对象那比较结果为真。
  5. **不带缓存的channels** 的发送/接受操作会使自己阻塞直到另一个goroutine被接受/已发送, 所以又叫同步channels
  6. **带缓存的channels** 内部有一个元素队列， 发送操作就是向内部缓存队列的尾部插入元素，接收操作则是从队列的头部删除元素。 如果队列已满，那么就会像无缓存channels一样阻塞。
  7. 多个goroutines并发地向同一个channel发送数据，或从同一个channel接收数据时，如果我们使用了无缓存的channel，那么慢的goroutines将会因为没有人接收而被永远卡住。这种情况，称为goroutines泄漏，这将是一个BUG。*和垃圾变量不同，泄漏的goroutines并不会被自动回收，因此确保每个不再需要的goroutine能正常退出是重要的。*
  8. sync.WaitGroup可以用来计数活跃的goroutine


- **基于select的多路复用**
- **Goroutine的退出**
  
  用*关闭一个channel*来进行广播
  ```golang
  var done = make(chan struct{})

  func cancelled() bool {
    select {
    case <-done:
        return true
    default:
        return false
    }
  }

  // Cancel traversal when input is detected.
  go func() {
    os.Stdin.Read(make([]byte, 1)) // read a single byte
    close(done)
  }()
  for {
    select {
    case <-done:
        // Drain fileSizes to allow existing goroutines to finish.
        for range fileSizes {
            // Do nothing.
        }
        return
    case size, ok := <-fileSizes:
        // ...
    }
  }
  //轮询取消状态
  func walkDir(dir string, n *sync.WaitGroup, fileSizes chan<- int64) {
    defer n.Done()
    if cancelled() {
        return
    }
    for _, entry := range dirents(dir) {
        // ...
    }
  }

  ```

- **sync.Mutex互斥锁**
    
在Lock和Unlock之间的代码段中的内容goroutine可以随便读取或者修改，这个代码段叫做临界区。
    
    ```golang
    func Balance() int {
        mu.Lock()
        defer mu.Unlock()
        return balance
    }
    ```
- **sync.RWMutex读写锁**
  
读操作并行执行，但写操作会完全互斥。这种锁叫作“多读单写”锁（multiple readers, single writer lock）, RLock只能在临界区共享变量没有任何写入操作时可用。
```golang
var mu sync.RWMutex
var balance int
func Balance() int {
    mu.RLock() // readers lock
    defer mu.RUnlock()
    return balance
}
```
RWMutex只有当获得锁的大部分goroutine都是读操作，而锁在竞争条件下，也就是说，goroutine们必须等待才能获取到锁的时候，RWMutex才是最能带来好处的。


##### Go scheduler
协程:

协程拥有自己的寄存器上下文和栈。协程调度切换时，将寄存器上下文和栈保存到其他地方，在切回来的时候，恢复先前保存的寄存器上下文和栈。 因此，协程能保留上一次调用时的状态（即所有局部状态的一个特定组合），每次过程重入时，就相当于进入上一次调用的状态，换种说法：进入上一次离开时所处逻辑流的位置。 线程和进程的操作是由程序触发系统接口，最后的执行者是系统；协程的操作执行者则是用户自身程序，goroutine也是协程。

groutine能拥有强大的并发实现是通过**GPM调度模型**实现.

Go的调度器内部有四个重要的结构：M，P，S，Sched.

**M**: M代表内核级线程，一个M就是一个线程，goroutine就是跑在M之上的；M是一个很大的结构，里面维护小对象内存cache（mcache）、当前执行的goroutine、随机数发生器等等非常多的信息

**G**: 代表一个goroutine，它有自己的栈，instruction pointer和其他信息（正在等待的channel等等），用于调度。

**P**: P全称是Processor，处理器，它的主要用途就是用来执行goroutine的，所以它也维护了一个goroutine队列，里面存储了所有需要它来执行的goroutine, P用于调度的上下文。你可以把它看成一个本地化版本的调度器.

**Sched**：代表调度器，它维护有存储M和G的队列以及调度器的一些状态信息等。

**调度实现:**

有2个物理线程M，每一个M都拥有一个处理器P，每一个也都有一个正在运行的goroutine。P的数量可以通过GOMAXPROCS()来设置，它其实也就代表了真正的并发度，即有多少个goroutine可以同时运行。

P维护着这个队列（称之为runqueue），Go语言里，启动一个goroutine很容易：go function 就行，所以每有一个go语句被执行，runqueue队列就在其末尾加入一个goroutine，在下一个调度点，就从runqueue中取出（如何决定取哪个goroutine？）一个goroutine执行。

当一个OS线程M0陷入阻塞时，P转而在运行M1. 当MO返回时，它必须尝试取得一个P来运行goroutine，一般情况下，它会从其他的OS线程那里拿一个P过来， 如果没有拿到的话，它就把goroutine放在一个global runqueue里，然后自己睡眠（放入线程缓存里）。所有的P也会周期性的检查global runqueue并运行其中的goroutine，否则global runqueue上的goroutine永远无法执行。

另一种情况是P所分配的任务G很快就执行完了（分配不均），这就导致了这个处理器P很忙，但是其他的P还有任务，此时如果global runqueue没有任务G了，那么P不得不从其他的P里拿一些G来执行。

通常来说，如果P从其他的P那里要拿任务的话，一般就拿run queue的一半，这就确保了每个OS线程都能充分的使用。

----------

### Redis ###

**Redis的数据类型：** 
- String 整数，浮点数或者字符串
- Set 集合
- Zset 有序集合
- Hash 散列表
- List 列表

**Redis 为什么这么快?**
- 数据结构简单，操作省时
- 跑在内存上
- 多路复用io阻塞机制
  (尽量减少网络 IO 的时间消耗）

非阻塞 IO 内部实现采用 epoll，采用了 epoll+自己实现的简单的事件框架。epoll 中的读、写、关闭、连接都转化成了事件，然后利用 epoll 的多路复用特性，绝不在 io 上浪费一点时间。

- 单线程保证了系统没有线程的上下文切换

使用单线程，可以避免不必要的上下文切换和竞争条件，没有多进程或多线程引起的切换和 CPU 的消耗，不必考虑各种锁的问题，没有锁释放或锁定操作，不会因死锁而降低性能；

**常用的使用场景：**
- 缓存：提升服务器性能方面非常有效
- 排行榜，利用Redis的SortSet(有序集合)数据结构能够简单的搞定
- 计算器/限速器：利用Redis原子性的自增操作，我们可以统计类似用户点赞数、用户访问数等，这类操作如果用MySQL，频繁的读写会带来相当大的压力; 限速器比较典型的使用场景是限制某个用户访问某个API的频率，常用的有秒杀时，防止用户疯狂点击带来不必要的压力;
- 简单消息队列: 除了Redis自身的发布/订阅模式，我们也可以利用List来实现一个队列机制，比如：到货通知、邮件发送之类的需求，不需要高可靠，但是会带来非常大的DB压力，完全可以用List来完成异步解耦；
- Session共享: 以PHP为例，默认Session是保存在服务器的文件中，如果是集群服务，同一个用户过来可能落在不同机器上，这就会导致用户频繁登陆；采用Redis保存Session后，无论用户落在那台机器上都能够获取到对应的Session信息。

- 一些频繁被访问的数据: 经常被访问的数据如果放在关系型数据库，每次查询的开销都会很大，而放在redis中，因为redis是放在内存中的可以很高效的访问

**Redis的Master-slave 模式**

链接过程： 

1. 主服务器创建快照文件，发送给从服务器，并在发送期间使用缓冲区记录执行的写命令。快照文件发送完毕之后，开始向从服务器发送存储在缓冲区中的写命令；
2. 从服务器丢弃所有旧数据，载入主服务器发来的快照文件，之后从服务器开始接受主服务器发来的写命令；
3. 主服务器每执行一次写命令，就向从服务器发送相同的写命令。

主从链：

随着负载不断上升，主服务器可能无法很快地更新所有从服务器，或者重新连接和重新同步从服务器将导致系统超载。为了解决这个问题，可以创建一个中间层来分担主服务器的复制工作。中间层的服务器是最上层服务器的从服务器，又是最下层服务器的主服务器。

*Sentinel（哨兵)* 可以监听集群中的服务器，并在主服务器进入下线状态时，自动从从服务器中选举出新的主服务器。

----------


### MIT 6.824 Distributed Systems Spring 2020

link: [Videos](https://www.bilibili.com/video/BV1x7411M7Sf?from=search&seid=15797605702990137477)

- [X] 课程简介
- [X] RPC与多线程
- [X] GFS
- [X] Primary-Backup Replication
- [X] Go Threads and Raft
- [ ] Fault Tolerance - Raft

- 相关必读资料： 
  
  [Course website](https://pdos.csail.mit.edu/6.824/)
  
  [Guide to Raft](https://thesquareplanet.com/blog/students-guide-to-raft/)
  
  Lab1: MapReduce ----- [paper](https://pdos.csail.mit.edu/6.824/papers/mapreduce.pdf)
  
  Lab2: [Raft](https://pdos.csail.mit.edu/6.824/labs/lab-raft.html)-----[paper](https://pdos.csail.mit.edu/6.824/papers/raft-extended.pdf)
  
  Lab3:
  
  Lab4:
  
 - 未完待续
   
   
   
----------

### Interview Questions

 **1. mysql索引为什么要用B+树？**
 - 高度矮, 磁盘IO相对少
 - 非叶子节点只保存索引，不保存实际的数据，数据都保存在叶子节点中
 - 内部节点更小，一次IO可查更多关键词
 - B+树只需要去遍历叶子节点就可以实现整棵树的遍历， 提升范围查找效率
 - 每次查找都从根部到叶子，性能稳定

**2. 死锁4必要条件及预防处理?**
   [参考资料](https://blog.csdn.net/wenlijunliujuan/article/details/79614019)
 - 互斥条件  进程对资源进行排他性控制
 - 不可剥夺条件  进程所获得的资源只能是主动释放
 - 请求与保持条件  
   进程已经保持了至少一个资源，提出了新的资源请求，而该资源已被其他进程占有，此时请求进程被阻塞，但对自己已获得的资源保持不放。
 - 循环等待条件
    存在一种进程资源的循环等待链，链中每一个进程已获得的资源同时被 链中下一个进程所请求。

**3. Race Condition ?**
  两个进程同时试图修改一个共享内存的内容，在没有并发控制的情况下，最后的结果依赖于两个进程的执行顺序与时机。
	

- 解决原则：
- 不会有两个及以上进程同时出现在他们的critical section。 
- 不要做任何关于CPU速度和数量的假设。 
- 任何进程在运行到critical section之外时都不能阻塞其他进程。 
- 不会有进程永远等在critical section之前。

**4. 传输层协议 TCP/UDP**

 - TCP是面向连接的，可靠的流协议。TCP可实行“顺序控制”， “重发控制”， “流量控制”， “拥塞控制”。
 - UDP不可靠数据报协议，可以确保发送消息的大小，但是不保证数据一定送达，所以有时候需要重发。
 -  UDP主要用于哪些对高速传输和实时性有较高要求的通信或广播通信。
 -  TCP 可以在 IP 这种无连接的网络上也能够实现高可靠性的通信（ 主要通过检验和、序列号、确认应答、重发控制、连接管理以及窗口控制等机制实现）
- TCP 通信开始前需要做好连接准备，三次握手连接，四次挥手断开。
  (1)客户端：SYN=1 seq =j
  
  (2)服务器 SYN=1 ACK=1 ack=J+1,seq=K
  
  (3)客户端 ACK=1 ack=K+1    
- 利用窗口控制提高速度：TCP 以1个段为单位，每发送一个段进行一次确认应答的处理。这样的传输方式包的通信性能会比较低。TCP 引入了窗口这个概念。确认应答以更大的单位进行确认，转发时间将会被大幅地缩短。
- 在整个窗口的确认应答没有到达之前，如果其中部分数据出现丢包，那么发送端仍然要负责重传。为此，发送端主机需要设置缓存保留这些待被重传的数据，直到收到他们的确认应答。而收到确认应答的情况下，将窗口滑动到确认应答中的序列号的位置。这样可以顺序地将多个段同时发送提高通信性能。这种机制也别称为滑动窗口控制。

**5. Python全局解释器锁**

GIL Global Interpreter Lock.

官方解释：GIL is a mutex that protects access to Python objects, preventing multiple threads from executing Python bytecodes at once. This lock is necessary mainly because CPython's memory management is not thread-safe. (However, since the GIL exists, other features have grown to depend on the guarantees that it enforces.)

在多线程编程时，为了防止多个线程同时操作一个变量时发生冲突，我们会设置一个互斥锁，只有获取到这个锁的线程才可以操作这个变量，这样做虽然安全了，但是并行变串行影响了程序的效率。而GIL是Python解释器为了程序的稳定性，在解释多线程的程序时加一把全局解释锁，保证同一时刻只有一个线程在被解释，效率自然也就变低了。

GIL不是Python的特性，它是Python的C解释器在实现的时候引入的特性，不是说我们的Python代码写出来就自带了GIL，而是在执行时，CPython解释器在解释多线程程序时会受到GIL锁的影响。 

*如果用到了多线程编程，但是对并行没有要求，只是对并发有要求，那么GIL锁影响不大，如果对并行要求高，那么可以用multiprocess（多进程）替代Thread，这样每一个Python进程都有自己的Python解释器和内存空间。*

**6. 互斥锁，自旋锁**

互斥锁是为了对临界区加以保护，以使任意时刻只有一个线程能够执行临界区的代码，实现了多线程对临界资源的互斥访问。

互斥锁得不到锁时，线程会进入休眠，引发任务上下文切换，任务切换涉及一系列耗时的操作，因此用互斥锁一旦遇到阻塞切换，代价是十分昂贵的。

而自旋锁阻塞后不会引发上下文切换，当锁被其他线程占有时，获取锁的线程便会进入自旋，不断检测自旋锁的状态，直到得到锁，*所以自旋就是循环等待的意思*。

自旋锁使用与临界区代码比较短，锁的持有时间比较短的情况，否则会让其他线程一直等待造成饥饿现象。

**7. 缓存穿透，缓存雪崩，缓存击穿**

**缓存穿透**：访问一个不存在的key，缓存不起作用，请求会穿透到DB，流量大时DB会挂掉。

解决方案： 
  - 布隆过滤器，使用一个足够大的bitmap，用于存储可能访问的key，不存在的key直接被过滤；
  - 访问key未在DB查询到值，也将空值写进缓存，但可以设置较短过期时间。

**缓存雪崩**：大量的key设置了相同的过期时间，导致在缓存在同一时刻全部失效，造成瞬时DB请求量大、压力骤增，引起雪崩效应。

解决方案：通过给缓存设置过期时间时加上一个随机值时间，使得每个key的过期时间分布开来，不会集中在同一时刻失效。

**缓存击穿**： 一个存在的key，在缓存过期的一刻，同时有大量的请求，这些请求都会击穿到DB，造成瞬时DB请求量大、压力骤增。

解决方案： 在访问key之前，采用SETNX（set if not exists）来设置另一个短期key来锁住当前key的访问，访问结束再删除该短期key。

**8. **

----------

### Reading List

书名  | 阅读进度
------------- | -------------
[The Go Programing language](https://books.studygolang.com/gopl-zh/ch1/ch1-01.html)  |  275/374
[Effective Go](https://bingohuang.gitbooks.io/effective-go-zh-en/content/)  |  0/114
Go 语言高并发和微服务实战  |  100/390
超大流量分布式系统架构解决方案 | 220/220 done
Kubernetes 即学即用  | 80/218
机器学习应用系统设计 | 241/241 done
Linux/UNIX 编程手册 | 120/1176
深入理解计算机系统  |  435/733
剑指offer  |  196/333

----------


### Support or Contact

wechart: luopengllpp
email: luopengllpp@hotmail.com