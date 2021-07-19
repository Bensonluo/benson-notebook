# Main notebook

## Reading List

| 书名                                                         | 目前优先级 | 状态        | 阅读进度 |
| :----------------------------------------------------------- | ---------- | ----------- | -------- |
| [The Go Programing language](https://books.studygolang.com/gopl-zh/ch1/ch1-01.html) |            | Main points | 275/374  |
| [Effective Go](https://bingohuang.gitbooks.io/effective-go-zh-en/content/) |            | Completed   | 114/114  |
| Go 语言高并发和微服务实战                                    | review | Main points | 320/390 |
| 超大流量分布式系统架构解决方案                               |            | Completed   | 220/220  |
| Go语言实战                                                   | 4          | In progress | 125/246 |
| 分布式系统-常用技术及案例分析 - JAVA                         | 1          | In progress | 102/540 |
| 企业级大数据平台构建-架构与实现                              | 3 | OnHold      | 0/249    |
| 设计模式-Reusable O-O Software                               | 2          | In progress | 206/290  |
| Kubernetes 即学即用                                          |            | Main points | 80/218   |
| 机器学习应用系统设计                                         |            | Completed   | 241/241  |
| Linux/UNIX 编程手册                                          |            | OnHold      | 120/1176 |
| 深入理解计算机系统                                           |            | Main points | 435/733  |
| 剑指offer                                                    |           | Main points | 266/333  |
| Effective Python                                             |            | Completed   | 213/213  |
| 计算机网络 A top-down Approcach                              | 2          | In progress | 152/510  |
| Spring Boot in Action                                        | 999        | Blocked     | 0%       |
| Spring Microservice in Action                                | 999        | Blocked     | 0        |
| Spring in Action                                             | 999        | Blocked     | 0/464    |



------



## LeetCode Progress

[My Leetcode account](https://leetcode-cn.com/u/peng-194/)

| Problem Solved : | 135  |
| ---------------- | ---- |
| 简单             | 84   |
| 中等             | 49   |
| 困难             | 2    |
| 总提交数         | 269  |





----------



## [GO电商网站高并发秒杀项目实践](https://bensonluo.github.io/benson-notebook/go-shopping-practice/)



不断更新中

点击标题跳转



----------



## [GO电商微服务容器化项目开发运维实践-DevOps](https://bensonluo.github.io/benson-notebook/go-microservice-devops/)



不断更新中

点击标题跳转



----------



## [Data structure and Algorithm](https://bensonluo.github.io/benson-notebook/algorithms/) 

点击标题跳转

### 各数据结构时间复杂度

| 数据结构   | 插入    | 删除    | 查找    |
| :--------- | ------- | ------- | ------- |
| 数组       | o(n)    | o(1)    | o(n)    |
| 有序数组   | o(logn) | o(n)    | o(n)    |
| 链表       | o(n)    | o(1)    | o(n)    |
| 有序链表   | o(n)    | o(n)    | o(n)    |
| 二叉树最坏 | o(n)    | o(n)    | o(n)    |
| 二叉树一般 | o(logn) | o(logn) | o(logn) |
| 平衡树AVL  | o(logn) | o(logn) | o(logn) |
| 哈希表     | o(1)    | o(1)    | o(1)    |
| 双向链表   | O(n)    | O(1)    | O(1)    |

### 主要排序算法的效率

| 排序算法 | 平均时间   | 最坏时间   | 空间     | 稳定性 | 注释 |
| -------- | ---------- | ---------- | -------- | ------ | ---- |
| 冒泡     | o(n^2)     | o(n^2)     | o(1)     | 稳定   |      |
| 选择     | o(n^2)     | o(n^2)     | o(1)     | 不稳定 |      |
| 插入     | o(n^2)     | o(n^2)     | o(1)     | 稳定   |      |
| 快速     | o(n log n) | o(n^2)     | o(log n) | 不稳定 |      |
| 归并     | o(n log n) | o(n log n) | o(n)     | 稳定   |      |


----------

## [Golang 基础](https://bensonluo.github.io/benson-notebook/golang/)

点击标题跳转


----------


## Networking


----------


## Database

| 特性           | InnoDB | MyISAM | MEMORY |
| -------------- | ------ | ------ | ------ |
| 事物安全       | 支持   | 不支持 | 不支持 |
| 对外键的支持   | 支持   | 不支持 | 不支持 |
| 存储限制       | 64T    | 有     | 有     |
| 空间使用       | 高     | 低     | 低     |
| 内存使用       | 高     | 低     | 高     |
| 插入数据的速度 | 低     | 高     | 高     |

InnoDB是Mysql的默认存储引擎(5.5.5之前是MyISAM）

当需要使用数据库事务时候，InnoDb是首选

由于锁的粒度小，写操作不会锁定全表。所以在并发度较高的场景下使用会提升效率的。

大批量的插入语句时（INSERT语句）在MyIASM引擎中执行的比较的快，但是UPDATE语句在Innodb下执行的会比较的快，尤其是在并发量大的时候。



----------

## Redis ##

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

**分布式锁**

分布式锁一般有如下的特点：

- 互斥性： 同一时刻只能有一个线程持有锁
- 可重入性： 同一节点上的同一个线程如果获取了锁之后能够再次获取锁
- 锁超时：和J.U.C中的锁一样支持锁超时，防止死锁
- 高性能和高可用： 加锁和解锁需要高效，同时也需要保证高可用，防止分布式锁失效
- 具备阻塞和非阻塞性：能够及时从阻塞状态中被唤醒

我们一般实现分布式锁有以下几种方式实现分布式锁：

- 基于数据库
- 基于Redis
- 基于zookeeper

**cluster怎么保证键的均匀分配？为什么用Crc16算法，和MD5的区别？**

Redis Cluser采用虚拟槽分区，所有的键根据哈希函数映射到0~16383个整数槽内，计算公式：slot=CRC16（key）&16383。

CRC  的信息字段和校验字段的长度可以选定。

Redis 采用的是基于字节查表法的CRC校验码生成算法，计算效率和速度比MD5快，且取得了速度和空间占用的平衡。



--------



## System Design 

**秒杀红包系统**：（超高并发，限流，削峰，维持可用）

1. 业务上 限流（分散时间，区别用户，点击门槛）

2. 技术上 抗压 
   - 监控如达到压力测试的极限QPS, 直接返回已抢完
   
   - 提高服务器数量性能
   
   - 分层效验：读可弱一致性效验，写强一致性
   
   - 用消息队列缓冲请求
   
     

---------



## MIT 6.824 Distributed Systems

link: [Videos](https://www.bilibili.com/video/BV1x7411M7Sf?from=search&seid=15797605702990137477)

- [X] 课程简介
- [X] RPC与多线程
- [X] GFS
- [X] Primary-Backup Replication
- [X] Go Threads and Raft
- [x] Fault Tolerance - Raft

- 相关必读资料： 
  
  [Course website](https://pdos.csail.mit.edu/6.824/)
  
  [Guide to Raft](https://thesquareplanet.com/blog/students-guide-to-raft/)
  
  Lab1: MapReduce ----- [paper](https://pdos.csail.mit.edu/6.824/papers/mapreduce.pdf)
  
  Lab2: [Raft](https://pdos.csail.mit.edu/6.824/labs/lab-raft.html)-----[paper](https://pdos.csail.mit.edu/6.824/papers/raft-extended.pdf)
  
  Lab3: [Fault-tolerant Key/Value Service](https://pdos.csail.mit.edu/6.824/labs/lab-kvraft.html)
  
  Lab4: [Sharded Key/Value Service](https://pdos.csail.mit.edu/6.824/labs/lab-shard.html)
  
  
----------



## Interview Questions

AKA 八股文 以及其他未能及时归类

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

**7. Redis缓存穿透，缓存雪崩，缓存击穿**

- **缓存穿透**：访问一个不存在的key，缓存不起作用，请求会穿透到DB，流量大时DB会挂掉。

  解决方案： 
  - 布隆过滤器，使用一个足够大的bitmap，用于存储可能访问的key，不存在的key直接被过滤；
  - 访问key未在DB查询到值，也将空值写进缓存，但可以设置较短过期时间。

- **缓存雪崩**：大量的key设置了相同的过期时间，导致在缓存在同一时刻全部失效，造成瞬时DB请求量大、压力骤增，引起雪崩效应。

   解决方案：通过给缓存设置过期时间时加上一个随机值时间，使得每个key的过期时间分布开来，不会集中在同一时刻失效。

- **缓存击穿**： 一个存在的key，在缓存过期的一刻，同时有大量的请求，这些请求都会击穿到DB，造成瞬时DB请求量大、压力骤增。

  解决方案： 在访问key之前，采用SETNX（set if not exists）来设置另一个短期key来锁住当前key的访问，访问结束再删除该短期key。

**8. 海量数据的查询优化？**

- 优化索引： 通过建立合理高效的索引,提高查询的速度.
- SQL优化： 组织优化SQL语句,使查询效率达到最优,在很多情况下要考虑索引的作用.
- 水平拆表： 如果表中的数据呈现出某一类特性,比如呈现时间，地点等,那么可以根据特殊性将表拆分成多个， 进行拆分查询后再合并结果
- 垂直拆表： 单表拆多表，将常用和不常用，长内容和短内容分开 （由于数据库每次查询都是以块为单位，而每块的容量是有限的，通常是十几K或几十K，将表按字段拆分后，单次IO所能检索到的行数通常会提高很多，查询效率就能提高上去。）
- 建立中间表，以空间换时间
- 用内存缓存数据， 比如使用Redis

**9. Typescript 的优势**

1. TypeScript 是强类型面对对象编程语言, 增加了代码的可读性和可维护性

2. 支持静态类型，支持 Class、Interface、Generics、Enums等。

3. TypeScript 拥抱了 ES6 规范

4. 兼容很多第三方库。

5. TypeScript 在开发时就能给出编译错误，而 JavaScript 错误则需要在运行时才能暴露

   

----------



## Support or Contact


email: luopengllpp@hotmail.com