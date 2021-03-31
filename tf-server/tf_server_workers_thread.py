#!/usr/bin/env python3

"""
Adapted from:
https://docs.python.org/3/library/asyncio-protocol.html#tcp-echo-server
https://www.oreilly.com/library/view/using-asyncio-in/9781492075325/ch04.html


A server with a queue and 
"""

import asyncio
import json
#import struct
import numpy as np
import concurrent.futures
from premsel_test_formirek_multi import Network
from graph_data import GraphData

BATCH_SIZE = 100
NUM_WORKERS = 8
WAIT_FOR = 1e-2
SERVER_IP = '127.0.0.1'
SERVER_PORT = 8888
STREAM_LIMIT = 128 * 1024 * 1024  # stream reader limit 128MB

queue = asyncio.Queue()

async def client(reader, writer):
    try:
        while True:
            msg = await reader.readuntil(b'\0')
            #addr = writer.get_extra_info('peername')

            await queue.put((writer, msg))
    except asyncio.CancelledError:
        #Remote closing connection
        pass
    except asyncio.IncompleteReadError:
        #Remote disconnected
        pass
    finally:
        #Remote closed
        writer.close()
        await writer.wait_closed()

def eval_msg(msg):
    message = json.loads(msg[:-1].decode())

    num_outputs = sum(message['ini_clauses'])

    #print(f"Num of outputs: {num_outputs!r}")

    out_msg = num_outputs.to_bytes(4, byteorder = 'little')

    #for i in range(num_outputs):
    #    print(f"{i / num_outputs!r}")
    #    out_msg += struct.pack('<f', i / num_outputs) # little-endian

    evaluations = np.arange(num_outputs, dtype=np.float32)
    #print(f"Evaluations: {evaluations!r}")
    out_msg += evaluations.tobytes()
   
    return out_msg

def eval_msgs(msgs):
    messages = [GraphData(json.loads(msg[:-1].decode())) for msg in msgs]
    return network.predict(batch)
#return [eval_msg(x) for x in msgs]

async def proc_writers_msgs(writers_msgs, pool):
    writers, msgs = zip(*writers_msgs)

    loop = asyncio.get_running_loop()
    out_msgs = await loop.run_in_executor(pool, eval_msgs, msgs)

    for writer, out_msg in zip(writers, out_msgs):
        writer.write(out_msg)
        await writer.drain()

        ##print("Close the connection")
        #writer.close()
        #await writer.wait_closed()

async def worker(i, pool):
    while True:
        if queue.qsize() < BATCH_SIZE:
            await asyncio.sleep(WAIT_FOR)
        if not queue.empty():
            writers_msgs = []

            for _ in range(BATCH_SIZE):
                try:
                    writers_msgs.append(queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            print(f'Worker: {i} Batch size: {len(writers_msgs)}')

            await proc_writers_msgs(writers_msgs, pool)

            for _ in range(len(writers_msgs)):
                queue.task_done()
    
async def main(*args, **kwargs):    
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
    #with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        workers = [asyncio.create_task(worker(i, pool)) for i in range(NUM_WORKERS)]
        server = await asyncio.start_server(*args, **kwargs)

        addr = server.sockets[0].getsockname()
        print(f'Serving on {addr}')

        async with server:
            await server.serve_forever()

        for w in workers:
            w.cancel()

        await asyncio.gather(*workers, return_exceptions=True)

network = Network()
network.load("../models/gnn/premsel_enigma_01_2020_T30_loop02_2")
asyncio.run(main(client,
                 host=SERVER_IP,
                 port=SERVER_PORT,
                 limit=STREAM_LIMIT))
