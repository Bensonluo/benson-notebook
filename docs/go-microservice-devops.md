# GO电商微服务容器化项目开发运维实践-DevOps



[代码根目录](https://github.com/Bensonluo/go-microservice-devops-sre)

## DDD: Domain Driven Design

- 不要数据驱动或界面驱动

- 微服务要界限清晰， 职能清晰的分层， 控制合适的粒度而不是过度的拆分

- 四层架构:  Interface -> Application -> Domain -> Infrastructure

![](https://github.com/Bensonluo/benson-notebook/blob/master/images/ddd.png)



## Micro 基础

gRPC server(service)   -> Proto Request (response) -> gRpc stub(client)

Protocal Buffers:  轻便高效的序列化结构化数据的协议

![protobuffer](https://github.com/Bensonluo/benson-notebook/blob/master/images/protob.png)



### go-micro框架

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

![](https://github.com/Bensonluo/benson-notebook/blob/master/images/zujiantu.png)



未完待续....

working on it!



![架构图](https://github.com/Bensonluo/benson-notebook/blob/master/images/zongjiagou.png)

微服务模块

注册配置中心

链路追踪

熔断，限流，负载均衡

监控能力完善

服务级观测台

k8s









