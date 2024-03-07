## Kafka Connect
- Data integration system and ecosystem
- Becuase some other systems are not kafka
- External client process; does not run on Brokers
- Horizontally scalable
- Fault tolerant

## Connectors
- Pluggable software component
- Interfaces to external systems and to kafka
- Also exist as runtime entities
- Source connectors act as Producers
- Sink connectors act as Consumers


![alt text](../images/inside-kafka-connect.jpg)


## Hands On
```shell
$ confluent login --save
$ confluent environment list
$ confluent environment use {env-id}
$ confluent kafka cluster list
$ confluent kafka cluster use {cluster-id}
$ confluent kafka topic create transactions
$ confluent kafka topic list
$ confluent connect plugin list
$ confluent connect create --config [datagen-source-config.json](https://github.com/confluentinc/learn-kafka-courses/blob/main/kafka-connect-101/datagen-source-config.json)
$ confluent connect list
$ confluent connect describe {ID}
$ confluent kafka topic consume -b transactions \
  --value-format avro \
  --api-key $CLOUD_KEY \
  --api-secret $CLOUD_SECRET \
  --sr-endpoint $SCHEMA_REGISTRY_URL \
  --sr-api-key <SR key> \
  --sr-api-secret <SR secret> 
$ confluent connect pause <connector ID>
$ confluent connect describe <connector ID>
$ confluent connect delete <connector ID>
```

### Deploying Kafka Connect
