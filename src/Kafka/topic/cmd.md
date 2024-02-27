## Create
```bash
$ bin/kafka-topics.sh --create --bootstrap-server localhost:9094 --topic kinaction_helloworld --partitions 3 --replication-factor 3
```
## List
```bash
$ bin/kafka-topics.sh --list --bootstrap-server localhost:9094
```
## Describe
```bash
$ ./bin/kafka-topics.sh --bootstrap-server localhost:9094 --describe --topic kinaction_helloworld
```

