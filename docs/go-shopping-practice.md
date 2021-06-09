# GO电商网站高并发秒杀项目实践



[代码根目录](https://github.com/Bensonluo/go-flash-sale)

## 需求分析

- 前端页面需要承载大流量

- 大并发状态要解决超卖问题
- 后端接口要可以方便的横向扩展



## 架构设计

CDN - > 流量负载 - > 流量拦截 - > 后端集群 - > RabbitMQ - > 队列消费 - > MySql



## RabbitMQ

简单模式/工作模式/发布订阅模式/路由模式/Topic模式

Code :  [Github](https://github.com/Bensonluo/go-flash-sale/tree/main/simple-rabbitmq)



## IRIS Framework

简单前端模板 -> Done

产品模块CURD API -> Done

订单模块CURD API -> Done



...未完待续

