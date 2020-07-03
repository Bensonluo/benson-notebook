## Benson's Personal Pages


### Data structure and Algorithm

[Leetcode account](https://leetcode-cn.com/u/peng-194/)


----------


### Go programing language
###### 变量声明：

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
###### Pointer
```golang
x := 1 
p := &x // p, of type *int, points to x (指针)
fmt.Println(*p) // "1"
*p = 2 // equivalent to x = 2 (指向变量内存地址的值)
fmt.Println(x) // "2"
```
###### 二元运算符
Go语言中关于算术运算、逻辑运算和比较运算的二元运算符，它们按照先级递减的顺序的排列：
```
* / % << >> & &^
+ - | ^
== != < <= > >=
&&
||
```

###### Slice
appendInt 函数
``` golang
func appendInt(x []int, y int) []int {
	var z []int
	zlen := len(x) + 1
	if zlen <= cap(x) {
		// There is room to grow. Extend the slice.
		z = x[:zlen]
	} else {
		// There is insufficient space. Allocate a new array.
		// Grow by doubling, for amortized linear complexity.
		zcap := zlen
		if zcap < 2*len(x) {
			zcap = 2 * len(x)
	}
	z = make([]int, zlen, zcap)
	copy(z, x) // a built-in function; see text
	}
	z[len(x)] = y
	return z
}
```
内置的append函数可能使用比appendInt更复杂的内存扩展策略。因此，通常我们并不知道append调用是否导致了内存的重新分配，因此我们也不能确认新的slice和原始的slice是否引用的是相同的底层数组空间。同样，我们不能确认在原先的slice上的操作是否会影响到新的slice。因此，通常是将append返回的结果直接赋值给输入的slice变量：runes = append(runes, r)

更新slice变量不仅对调用append函数是必要的，实际上对应任何可能导致长度、容量或底层数组变化的操作都是必要的。要正确地使用slice，需要记住尽管底层数组的元素是间接访问的，但是slice对应结构体本身的指针、长度和容量部分是直接访问的。要更新这些信息需要像上面例子那样一个显式的赋值
操作。从这个角度看，slice并不是一个纯粹的引用类型，它实际上是一个聚合类型。

- slice可以用来模拟stack
```golang
使用append函数将新的值压入stack：
stack = append(stack, v) // push v
stack的顶部位置对应slice的最后一个元素：
top := stack[len(stack)-1] // top of stack
通过收缩stack可以弹出栈顶的元素
stack = stack[:len(stack)-1] // pop
要删除slice中间的某个元素并保存原有的元素顺序，可以通过内置的copy函数将后面的子slice向前依次移动一位完成：
func remove(slice []int, i int) []int {
	copy(slice[i:], slice[i+1:])
	return slice[:len(slice)-1]
}
func main() {
	s := []int{5, 6, 7, 8, 9}
	fmt.Println(remove(s, 2)) // "[5 6 8 9]"
}
如果删除元素后不用保持原来顺序的话，我们可以简单的用最后一个元素覆盖被删除的元素：
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
```
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

###### Map, Dictionary, Hashmap 
``` golang
// 创建
m:=make(map[string]int)
// 设置kv
m["hello"]=1
// 删除k 失败返0
delete(m,"hello")
// 遍历
for k,v:=range m{
    println(k,v)
}
```
- map 元素不能取址操作，原因是map可能随着元素数量的增长而重新分配更大的内存空间，从而可能导致之前的地址无效
- map 键需要可比较，不能为 slice、map、function
- map 值都有默认值，可以直接操作默认值，如：m[age]++ 值由 0 变为 1
- 比较两个 map 需要遍历，其中的 kv 是否相同，因为有默认值关系，所以需要检查 val 和 ok 两个值

###### Marshaling
```
data, err := json.Marshal(movies)
data, err := json.MarshalIndent(movies, "", " ") //带缩进
if err != nil {
	log.Fatalf("JSON marshaling failed: %s", err)
}
fmt.Printf("%s\n", data)
```
  
###### 垃圾回收
Go语言的自动垃圾收集器从每个包级的变量和每个当前运行函数的每一个局部变量开始，通过指针或引用的访问路径遍历，是否可以找到该变量。如果不可达 -> 回收

注意：如果将指向短生命周期对象的指针保存到具有长生命周期的对象中，特别是保存到全局变量时，会阻止对短生命周期对象的垃圾回收（从而可能影响程序的性能）。



----------
### MIT 6.824 Distributed Systems Spring 2020

link: [Videos](https://www.bilibili.com/video/BV1x7411M7Sf?from=search&seid=15797605702990137477)


----------
### Reading List
[The Go Programing language](https://books.studygolang.com/gopl-zh/ch1/ch1-01.html)        进度 126/374


----------


### Support or Contact

wechart: luopengllpp
email: luopengllpp@hotmail.com