# Go programing language

##### Basic

- Variable declaration
``` golang
s := ""
var s string
var s = ""
var s string = ""
var identifier []type
slic := make([]int,len)
map_variable := make(map[key_data_type]value_data_type)
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

- string && byte && rune

  互相转换

  ```golang
  // string to []byte
      s1 := "string"
      by := []byte(s1)
  
      // []byte to string
      s2 := string(by)
  
  
  //string 转 rune
  r := []rune(str)
  
  //rune 转 string
  str = string(r)
  ```

  黑魔法转换 - 性能更优

  ```golang
  func String2Bytes(s string) []byte {
      sh := (*reflect.StringHeader)(unsafe.Pointer(&s))
      bh := reflect.SliceHeader{
          Data: sh.Data,
          Len:  sh.Len,
          Cap:  sh.Len,
      }
      return *(*[]byte)(unsafe.Pointer(&bh))
  }
  
  func Bytes2String(b []byte) string {
      return *(*string)(unsafe.Pointer(&b))
  }
  ```

  从go源码来看，string其实是一个指向byte数组的指针。

  字符串string是不可更改的，但是可以给他重新分配空间，给指针重新赋值。但是这也导致了他效率低下，因为之前的空间需要被gc回收。

  ```go
  s := "A1" // 分配存储"A1"的内存空间，s结构体里的str指针指向这快内存
  s = "A2"  // 重新给"A2"的分配内存空间，s结构体里的str指针指向这快内存
  ```

  而 []byte和string的差别是更改变量的时候array的内容可以被更改。

  ```go
  s := []byte{1} // 分配存储1数组的内存空间，s结构体的array指针指向这个数组。
  s = []byte{2}  // 将byte array的内容改为2
  ```

  ```golang
  // rune能表示的范围更多，比如中文(占3个字符)
  	str2:="你好,中国"
  	c:=[]rune(str2)
  	d:=[]byte(str2)
  	//c: [20320 22909 44 20013 22269]  中文字符也能拆
  	fmt.Println("c:",c) 
  	//d: [228 189 160 229 165 189 44 228 184 173 229 155 189]  一个中文拆成3个字符表示4*3+1=13
  	fmt.Println("d:",d) 
  }
  ```

  ```golang
  //编辑修改字符串string 最好用rune 因为一个 UTF8 编码的字符可能会占多个字节
  x := "text"
  xRunes := []rune(x)
  xRunes[0] = '人'
  x = string(xRunes)
  fmt.Println(x)    // 人ext
  ```

  总结：

  - byte 等同于int8，常用来处理ascii字符
  - rune 等同于int32，常用来处理unicode或utf-8字符

  - string可以直接比较，而[]byte不可以，所以[]byte不可以当map的key值。

  - 因为无法修改string中的某个字符，需要粒度小到操作一个字符时，用[]byte。
  - string值不可为nil，所以如果你想要通过返回nil表达额外的含义，就用[]byte。
  - []byte切片这么灵活，想要用切片的特性就用[]byte。
  - 需要大量字符串处理的时候用[]byte，性能好很多。

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

Slice的删除by Index

`seq = append(seq[:index], seq[index+1:]...)`

插入：

```golang
rear:=append([]string{},ss[index:]...) 创建临时切片保存后部元素
ss=append(ss[0:index],"inserted") 追加到前切片尾部
ss=append(ss,rear...) 合并
```

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
queue := make([]int,0)
// enqueue入队
queue=append(queue,10)
// dequeue出队
v := queue[0]
queue = queue[1:]
// 长度0为空
len(queue)==0
```

##### Hashmap 
``` golang
// 创建
map := make(map[string]int)
map := map[string]int{}
// 设置kv
map["key"] = 1
// 删除k 失败返0
delete(map,"key")
// 遍历
for k,v := range map{
    println(k,v)
}
```
- 
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

流程图：

![](https://pic1.zhimg.com/80/v2-3fd461ae369acf7f71a1cb055a4f5154_1440w.jpg)

重点概念：

- **写屏障**
- **gray black white三色标记**

1. 黑色: 对象在这次GC中已标记,且这个对象包含的子对象也已标记
2. 灰色: 对象在这次GC中已标记, 但这个对象包含的子对象未标记
3. 白色: 对象在这次GC中未标记

- **gc-root 可达性分析**

- **并行的标记/扫描**

- **STW 停止**世界（暂停用户协程）/启动用户协程

  

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

groutine能拥有强大的并发实现是通过MPG调度模型**实现.

Go的调度器内部有四个重要的结构：M，P，G，Sched.

**M**: M代表内核级线程，一个M就是一个线程，goroutine就是跑在M之上的；M是一个很大的结构，里面维护小对象内存cache（mcache）、当前执行的goroutine、随机数发生器等等非常多的信息

**P**: P全称是Processor，处理器，它的主要用途就是用来执行goroutine的，所以它也维护了一个goroutine队列，里面存储了所有需要它来执行的goroutine, P用于调度的上下文。你可以把它看成一个本地化版本的调度器.

**G**: 代表一个goroutine，它有自己的栈，instruction pointer和其他信息（正在等待的channel等等），用于调度。

**Sched**：代表调度器，它维护有存储M和G的队列以及调度器的一些状态信息等。

**调度实现:**

有2个物理线程M，每一个M都拥有一个处理器P，每一个P也都有一个正在运行的goroutine。P的数量可以通过GOMAXPROCS()来设置，它其实也就代表了真正的并发度，即有多少个goroutine可以同时运行。

M会和一个系统内核线程绑定，而P和G的关系是一对多，M与P, P与G的关系都是动态可变的。

在运行过程中， M与P的组合才能为G提供运行环境，多个可执行的G会挂在某个P上等待调度和执行，P由程序决定，M由Go语言创建。

M和P会适时组合与断开，假如某个G阻塞了M，P就会携等待执行的G队列转投新M.

##### Golang 规范与注意

- Golang主程序必须要等待所有的Goroutine结束才能够退出，否则如果先退出主程序会导致所有的Goroutine可能未执行结束就退出了， 用WaitGroup.
- 每个Goroutine都要有recover机制，因为当一个Goroutine抛panic的时候只有自身能够捕捉到其它Goroutine是没有办法捕捉的, 如果没有recover机制，整个进程会crash。
- Recover只能在defer里面生效，如果不是在defer里调用，会直接返回nil。
- Goroutine发生panic时，只会调用自身的defer，所以即便主Goroutine里写了recover逻辑，也无法recover。

```golang
package main

import (
    "sync"
    "fmt"
    "time"
)

func calc(w *sync.WaitGroup, i int)  {
    defer func() {
        err := recover()
        if err != nil {
          fmt.Println("panic error.")
        }
    }()
    
    fmt.Println("calc: ", i)
    time.Sleep(time.Second)
    w.Done()
}

func main()  {
    # WaitGroup能够一直等到所有的goroutine执行完成，并且阻塞主线程的执行，直到所有的goroutine执行完成。
    wg := sync.WaitGroup{}    
    for i:=0; i<10; i++ {
        wg.Add(1)
        go calc(&wg, i)
    }
    # 阻塞主线程等到所有的goroutine执行完成
    wg.Wait()
    fmt.Println("all goroutine finish")
}
```

- 使用 http.Client，如果没有 `resp.Body.Close()`，可能导致 goroutine 泄露。
- Slice -> reslice 的地址引用问题。
- 还有很多可以查阅 [50 Shades of Go: Traps, Gotchas, and Common Mistakes for New Golang Devs](http://devs.cloudimmunity.com/gotchas-and-common-mistakes-in-go-golang/)

