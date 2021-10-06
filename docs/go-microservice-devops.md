# GO micro service project DevOps



[github code](https://github.com/Bensonluo/go-microservice-devops-sre)

## DDD: Domain Driven Design

- 不要数据驱动或界面驱动

- 微服务要界限清晰， 职能清晰的分层， 控制合适的粒度而不是过度的拆分

- 四层架构:  Interface -> Application -> Domain -> Infrastructure

![](https://github.com/Bensonluo/benson-notebook/blob/master/images/ddd.png)



## Micro basics

gRPC server(service)   -> Proto Request (response) -> gRpc stub(client)

Protocal Buffers:  轻便高效的序列化结构化数据的协议

![protobuffer](https://github.com/Bensonluo/benson-notebook/blob/master/images/protob.png)



### go-micro framework

1. 提供分布式系统的核心库
2. 对分布式系统的高度抽象
3. 可插拔的架构，按需使用

##### 组件：

1. 注册：服务发现
2. 选择器：实现负载均衡
3. 传输：服务间通讯
4. Broker：提供异步通讯的消息发布/订阅接口
5. 编码Codec： 消息传输到两端的编码解码
6. Server, Client

![组件图](https://github.com/Bensonluo/benson-notebook/blob/master/images/zujiantu.png)

![架构图](https://github.com/Bensonluo/benson-notebook/blob/master/images/zongjiagou.png)



### 注册配置中心 Consul

服务网格解决方案： 

##### 关键功能：

服务发现/健康检查/

键值对存储：键值存储可用于任何目的—> 动态配置，功能标记，协调，领导者竞选

安全服务通信： Consul可为服务生成分发TLS证书，建立相应的TLS连接

支持多数据中心

##### Gossip Protocol

局域网Lan Pool

- Client 自动发现Server节点， 减少配置量
- 分布式故障检测 在几个server上执行
- 快速广播事件

广域网Wan Pool

- Wan pool 全局唯一
- 不同数据中心的server都会加入WAN pool
- 允许服务器执行跨数据中心请求

##### Raft Protocol： 选举协议



### 微服务链路追踪

Jaeger的作用： 监视和诊断基于微服务的分布式系统

主要特性：高扩展性,  原生支持OpenTracing, 可观测性

术语Span：

- Jaeger的逻辑工作单元

- 具有操作名称，开始时间和持续时间

- 跨度可以嵌套排序并建立因果关系模型





未完待续....

working on it!

熔断，限流，负载均衡

监控能力完善

服务级观测台

k8s









