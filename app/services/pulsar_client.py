"""
Apache Pulsar Client with Asyncio Bridge

Pulsar Python client is NOT async natively - we bridge with ThreadPoolExecutor
Corrected based on technical review to avoid await on blocking calls
"""

import pulsar
import json
import asyncio
import logging
from typing import Callable, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("darwin.pulsar")

# Thread pool for blocking Pulsar calls
_executor = ThreadPoolExecutor(max_workers=4)


class PulsarEventBus:
    """
    Apache Pulsar client with asyncio bridge
    
    Features:
    - Multi-tenancy nativo
    - Avro schema registry para versionamento
    - Geo-replication ready
    - 2.5x throughput vs Kafka
    """
    
    def __init__(self, service_url: str = "pulsar://localhost:6650"):
        self.service_url = service_url
        self.client: Optional[pulsar.Client] = None
        self._producers: Dict[str, pulsar.Producer] = {}
        self._consumers: Dict[str, pulsar.Consumer] = {}
        self._running = False
    
    async def connect(self):
        """Connect to Pulsar cluster"""
        loop = asyncio.get_running_loop()
        
        # Blocking call -> executor
        self.client = await loop.run_in_executor(
            _executor,
            lambda: pulsar.Client(
                self.service_url,
                operation_timeout_seconds=30,
                logger=logger
            )
        )
        
        self._running = True
        logger.info(f"✅ Connected to Pulsar: {self.service_url}")
    
    def _get_producer(self, topic: str) -> pulsar.Producer:
        """Get or create producer (sync call)"""
        if topic not in self._producers:
            self._producers[topic] = self.client.create_producer(
                f"persistent://darwin/default/{topic}",
                schema=pulsar.schema.BytesSchema(),  # Use Bytes for flexibility
                compression_type=pulsar.CompressionType.LZ4,
                batching_enabled=True,
                batching_max_publish_delay_ms=10
            )
        return self._producers[topic]
    
    async def publish(self, topic: str, message: Dict[str, Any]) -> Any:
        """
        Publish message async using callback bridge
        
        Args:
            topic: Pulsar topic name
            message: Dictionary to publish
            
        Returns:
            Message ID from Pulsar
        """
        if not self.client:
            raise RuntimeError("Not connected to Pulsar. Call connect() first.")
        
        loop = asyncio.get_running_loop()
        data = json.dumps(message).encode('utf-8')
        
        # Create future for callback
        fut = loop.create_future()
        
        def callback(res, msg_id):
            """Callback when message is sent"""
            if res == pulsar.Result.Ok:
                loop.call_soon_threadsafe(fut.set_result, msg_id)
            else:
                loop.call_soon_threadsafe(
                    fut.set_exception,
                    Exception(f"Pulsar send failed: {res}")
                )
        
        # Get producer and send async
        producer = self._get_producer(topic)
        producer.send_async(data, callback)
        
        msg_id = await fut
        logger.debug(f"📤 Published to {topic}: {msg_id}")
        return msg_id
    
    async def subscribe(
        self,
        topic: str,
        subscription: str,
        handler: Callable[[Dict[str, Any]], Any]
    ):
        """
        Subscribe to topic and process messages async
        
        Args:
            topic: Pulsar topic name
            subscription: Subscription name (shared across instances)
            handler: Async handler for messages
        """
        if not self.client:
            raise RuntimeError("Not connected to Pulsar. Call connect() first.")
        
        loop = asyncio.get_running_loop()
        
        # Create consumer (blocking -> executor)
        consumer = await loop.run_in_executor(
            _executor,
            lambda: self.client.subscribe(
                f"persistent://darwin/default/{topic}",
                subscription_name=subscription,
                schema=pulsar.schema.BytesSchema(),
                consumer_type=pulsar.ConsumerType.Shared,  # Load balanced
                receiver_queue_size=1000
            )
        )
        
        self._consumers[topic] = consumer
        logger.info(f"✅ Subscribed to {topic} (subscription={subscription})")
        
        # Message processing loop
        while self._running:
            try:
                # Receive message (blocking -> executor)
                msg = await loop.run_in_executor(_executor, consumer.receive)
                
                # Decode
                try:
                    payload = json.loads(msg.data().decode('utf-8'))
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in message: {msg.message_id()}")
                    consumer.negative_acknowledge(msg)
                    continue
                
                # Process with handler
                try:
                    result = handler(payload)
                    if asyncio.iscoroutine(result):
                        await result
                    
                    consumer.acknowledge(msg)
                    logger.debug(f"📥 Processed message from {topic}: {msg.message_id()}")
                    
                except Exception as e:
                    logger.error(f"Handler error for {topic}: {e}")
                    consumer.negative_acknowledge(msg)
                    
            except Exception as e:
                logger.error(f"Consumer error for {topic}: {e}")
                await asyncio.sleep(1)  # Back off on error
    
    async def disconnect(self):
        """Disconnect from Pulsar"""
        self._running = False
        
        # Close consumers
        for consumer in self._consumers.values():
            consumer.close()
        
        # Close producers
        for producer in self._producers.values():
            producer.close()
        
        # Close client
        if self.client:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(_executor, self.client.close)
        
        logger.info("✅ Disconnected from Pulsar")


# Singleton
_pulsar_client: Optional[PulsarEventBus] = None


def get_pulsar_client() -> PulsarEventBus:
    """Get or create Pulsar client singleton"""
    global _pulsar_client
    if _pulsar_client is None:
        _pulsar_client = PulsarEventBus()
    return _pulsar_client


# Topic definitions (persistent://darwin/default/{topic})
TOPICS = {
    "ml_training": "ml-training-requests",
    "continuous_learning": "user-interaction-events",
    "plugin_health": "plugin-health-events",
    "system_alerts": "system-alert-events",
    "kec_results": "kec-analysis-results",
}

