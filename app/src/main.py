# Copyright (c) 2022 Robert Bosch GmbH and Microsoft Corporation
#
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""A sample skeleton vehicle app."""

import asyncio
import json
import logging
import signal

from vehicle import Vehicle, vehicle  # type: ignore
from velocitas_sdk.util.log import (  # type: ignore
    get_opentelemetry_log_factory,
    get_opentelemetry_log_format,
)

from velocitas_sdk.vdb.reply import DataPointReply
from velocitas_sdk.vehicle_app import VehicleApp, subscribe_topic

# Configure the VehicleApp logger with the necessary log config and level.
logging.setLogRecordFactory(get_opentelemetry_log_factory())
logging.basicConfig(format=get_opentelemetry_log_format())
logging.getLogger().setLevel("DEBUG")
logger = logging.getLogger(__name__)

GET_SPEED_REQUEST_TOPIC = "sampleapp/getSpeed"
GET_SPEED_RESPONSE_TOPIC = "sampleapp/getSpeed/response"
DATABROKER_SUBSCRIPTION_TOPIC = "sampleapp/currentSpeed"


class SampleApp(VehicleApp):

    def __init__(self, vehicle_client: Vehicle):
        super().__init__()
        self.Vehicle = vehicle_client

    async def on_start(self):
        await self.Vehicle.Speed.subscribe(self.on_speed_change)

    async def on_speed_change(self, data: DataPointReply):
        vehicle_speed = data.get(self.Vehicle.Speed).value

        await self.publish_event(
            DATABROKER_SUBSCRIPTION_TOPIC,
            json.dumps({"speed": vehicle_speed}),
        )

    @subscribe_topic(GET_SPEED_REQUEST_TOPIC)
    async def on_get_speed_request_received(self, data: str) -> None:
        logger.debug(
            "PubSub event for the Topic: %s -> is received with the data: %s",
            GET_SPEED_REQUEST_TOPIC,
            data,
        )

        vehicle_speed = (await self.Vehicle.Speed.get()).value

        await self.publish_event(
            GET_SPEED_RESPONSE_TOPIC,
            json.dumps(
                {
                    "result": {
                        "status": 0,
                        "message": f"""Current Speed = {vehicle_speed}""",
                    },
                }
            ),
        )

    def test(self):
        print("Helloworld")


async def main():
    """Main function"""
    logger.info("Starting SampleApp...")
    # Constructing SampleApp and running it.
    vehicle_app = SampleApp(vehicle)
    await vehicle_app.run()


LOOP = asyncio.get_event_loop()
LOOP.add_signal_handler(signal.SIGTERM, LOOP.stop)
LOOP.run_until_complete(main())
LOOP.close()
