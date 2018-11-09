#!/bin/bash

wait_for_healthy()
{
  CONTAINER_ID=$1

  until [ "`docker inspect -f {{.State.Health.Status}} $CONTAINER_ID`" == "healthy" ]
    do
      echo "container not healthy. Sleeping 10 seconds ..."
      sleep 10
    done
  echo "container is healthy. Continuing on ..."
}
CONTAINER_ID=$1
wait_for_healthy $CONTAINER_ID
