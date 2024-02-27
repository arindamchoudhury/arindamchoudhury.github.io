$ tar -xzf kafka_2.13-3.6.1.tgz
$ mv kafka_2.13-3.6.1 ~/
$ cd ~/kafka_2.13-3.6.1
$ export PATH=$PATH:~/kafka_2.13-3.6.1/bin

$ cd ~/kafka_2.13-3.6.1
$ bin/zookeeper-server-start.sh config/zookeeper.properties

$ cd ~/kafka_2.13-3.6.1
$ cp config/server.properties config/server0.properties
$ cp config/server.properties config/server1.properties
$ cp config/server.properties config/server2.properties


$ vim config/server0.properties
 
broker.id=0
listeners=PLAINTEXT://localhost:9092
log.dirs= /tmp/kafkainaction/kafka-logs-0
 
$ vim config/server1.properties
 
broker.id=1
listeners=PLAINTEXT://localhost:9093
log.dirs= /tmp/kafkainaction/kafka-logs-1
 
$ vim config/server2.properties
broker.id=2
listeners=PLAINTEXT://localhost:9094
log.dirs= /tmp/kafkainaction/kafka-logs-2


$ cd ~/kafka_2.13-3.6.1
$ bin/kafka-server-start.sh config/server0.properties
$ bin/kafka-server-start.sh config/server1.properties
$ bin/kafka-server-start.sh config/server2.properties


