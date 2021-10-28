# GO high performance flash sale system



[github code](https://github.com/Bensonluo/go-flash-sale)



## Requirements analysis

- 前端页面需要承载大流量
- 大并发状态要解决超卖问题
- 后端接口要可以方便的横向扩展
- 提高网站性能，保护数据库



## System design

CDN - > 流量负载 - > 流量拦截，分布式权限验证-> 分布式数量控制 

-> 后端集群 - > RabbitMQ(异步下单) - > 队列消费(排队写入) - > MySql

**秒杀系统**：（超高并发，限流，削峰，维持可用）

1. 业务上 限流（分散时间，区别用户，点击门槛）

2. 技术上 抗压 
   - 监控如达到压力测试的极限QPS, 直接返回已抢完
   - 提高服务器数量性能
   - 分层效验：读可弱一致性效验，写强一致性
   - 用消息队列缓冲请求



## RabbitMQ

简单模式/工作模式/发布订阅模式/路由模式/Topic模式

Code :  [Github](https://github.com/Bensonluo/go-flash-sale/tree/main/simple-rabbitmq)



## IRIS Framework

简单前端模板 -> Done

用户注册登陆模块 -> Done

产品模块 -> Done

订单模块 -> Done

基本服务架构完成



## Performance optimization

- 前端页面静态化加CDN
- SLB - 流量负载均衡
- 分布式安全验证 - 流量拦截
- 秒杀数量控制 - 防止超卖，增加性能
- RabbitMQ - 消息队列, 防止Mysql爆库



**一致性hash算法:** 

用于快速定位资源，均匀分布 -> 分布式存储，分布缓存, 负载均衡



**Redis 存在的瓶颈:** 

单机QPS - 8W左右，集群QPS - 千万级 ，分布式 - 千万级

但是对于单个商品超高流量的情况，使用集群和分布式并不能解决问题





