# GO电商网站高并发秒杀项目实践



[代码根目录](https://github.com/Bensonluo/go-flash-sale)

## 需求分析

- 前端页面需要承载大流量
- 大并发状态要解决超卖问题
- 后端接口要可以方便的横向扩展
- 提高网站性能，保护数据库



## 架构设计

CDN - > 流量负载 - > 流量拦截，分布式权限验证-> 分布式数量控制 

- > 后端集群 - > RabbitMQ(异步下单) - > 队列消费(排队写入) - > MySql



## RabbitMQ

简单模式/工作模式/发布订阅模式/路由模式/Topic模式

Code :  [Github](https://github.com/Bensonluo/go-flash-sale/tree/main/simple-rabbitmq)



## IRIS Framework

简单前端模板 -> Done

用户注册登陆模块 -> Done

产品模块 -> Done

订单模块 -> Done

基本服务架构完成



### 高并发抢购优化

- 前端页面静态化加CDN
- SLB - 流量负载均衡
- 分布式安全验证 - 流量拦截
- 秒杀数量控制 - 防止超卖，增加性能
- RabbitMQ - 消息队列, 防止Mysql爆库



**一致性hash算法:** 

用于快速定位资源，均匀分布 -> 分布式存储，分布缓存, 负载均衡



