```bash
$ confluent login --save
$ confluent environment list
$ confluent environment use {env-id}
$ confluent kafka cluster list
$ confluent kafka cluster use {cluster-id}
$ confluent api-key create --resource {cluster-id}
$ confluent api-key use {API_KEY} --resource {cluster-id}
$ confluent kafka topic list
$ confluent kafka topic consume --from-beginning {topic-name}
$ confluent kafka topic produce poems --parse-key
$ confluent 
````