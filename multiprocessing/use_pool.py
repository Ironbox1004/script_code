# import time
#
# def fun2(args):
#     x = args[0]
#     y = args[1]
#
#     time.sleep(1)
#
#     return x - y
#
# def fun_pool():
#     from multiprocessing import Pool
#
#     cpu_worker_num = 3
#     process_args = [(1, 1), (9, 9), (4, 4), (3, 3), ]
#     start_time = time.time()
#     with Pool(cpu_worker_num) as p:
#         outputs = p.map(fun2, process_args)
#     print(f'| outputs: {outputs}    TimeUsed: {time.time() - start_time:.1f}    \n')
#
#
# if __name__ =='__main__':
#     fun_pool()

import time

def func_pipe1(conn, p_id):
    print(p_id)

    time.sleep(0.1)
    conn.send(f'{p_id}_send1')
    print(p_id, 'send1')

    time.sleep(0.1)
    conn.send(f'{p_id}_send2')
    print(p_id, 'send2')

    time.sleep(0.1)
    rec = conn.recv()
    print(p_id, 'recv', rec)

    time.sleep(0.1)
    rec = conn.recv()
    print(p_id, 'recv', rec)


def func_pipe2(conn, p_id):
    print(p_id)

    time.sleep(0.1)
    conn.send(p_id)
    print(p_id, 'send')
    time.sleep(0.1)
    rec = conn.recv()
    print(p_id, 'recv', rec)


def run__pipe():
    from multiprocessing import Process, Pipe

    conn1, conn2 = Pipe()

    process = [Process(target=func_pipe1, args=(conn1, 'I1')),
               Process(target=func_pipe2, args=(conn2, 'I2')),
               Process(target=func_pipe2, args=(conn2, 'I3')), ]

    [p.start() for p in process]
    # print('| Main', 'send')
    # conn1.send(None)
    # print('| Main', conn2.recv())
    [p.join() for p in process]

if __name__ =='__main__':
    run__pipe()