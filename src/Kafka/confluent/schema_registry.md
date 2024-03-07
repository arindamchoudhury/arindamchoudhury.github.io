## Schema Registry
- Server process external to Kafka brokers
- Maintains a database of schemas
- HA deployment option available
- Consumer/Producer API component
- Defines schema compatibility rules per topic
- Producer API prevents incompatible messages from being produced
- Consumer API prevents incompatible messages from being consumed

### Supported Formats
- JSON Schema
- Avro
- Protocol Buffers

### HAnds On
````bash
confluent kafka topic consume --value-format avro --schema-registry-api-key {API KEY} --schema-registry-api-secret {API SECRET} orders
```