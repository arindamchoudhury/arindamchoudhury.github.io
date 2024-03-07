## Topics

- Named container for similar events
    - Systems contain lots of topics
    - Can duplicate data between topics
- Durable logs of events
    - Append only
    - Can only seek by offset, not indexed
- Events are immutable


### Topics are durable

## Partitioned topic
    - no key: > 2.4 - sticky partiion strategy, < 2.4 round robin 
    - key: hash function

```bash
$ confluent login --save
$ confluent environment list
$ confluent environment use {env-id}
$ confluent kafka cluster list
$ confluent kafka cluster use {cluster-id}
$ confluent api-key create --resource {cluster-id}
$ confluent api-key use {API_KEY} --resource {cluster-id}
$ confluent kafka topic list
$ confluent kafka topic describe poems
$ confluent kafka topic create --partitions 1 poems_1
$ confluent kafka topic create --partitions 4 poems_4
````