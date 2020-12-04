# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
Matmul operation for xilinx fpga
=========================
**Author**: `Insop Song <https://insop.github.io>`_

This operator is for calling xilinx fpga kernel for matmul.

"""

import tvm
from tvm import autotvm
from tvm import te

# from .utils import get_fp32_len
# from .. import generic, tag
from ..utils import traverse_inline, get_const_tuple

import tvm.testing
import numpy as np
import time

# TODO change name to dense

@tvm.register_func("tvm.contrib.xilinx_matmul_pynq")
def xilinx_matmul_pynq(a, b, c):

    pynq = True
    if pynq:
        import os
        import pynq
        from pynq import allocate


    # use call_cnt for lazy initialization
    if not hasattr(xilinx_matmul_pynq, 'call_cnt') and not hasattr(xilinx_matmul_pynq, 'ol'):
        xilinx_matmul_pynq.ol = None
        xilinx_matmul_pynq.call_cnt = 0

    xilinx_matmul_pynq.call_cnt += 1

    # lazy initialization, such as xclbin loading for fpga
    if xilinx_matmul_pynq.call_cnt == 1 and pynq:
        print("xclbin load")
        XCLBIN_FILE = "/data/Projects/transformer_simple/src/python/From_Ties/krnl_matmulbertl_int16.xclbin"
        xilinx_matmul_pynq.ol=pynq.Overlay(XCLBIN_FILE)

    start = time.time()
    # import pdb;pdb.set_trace()

    print("xilinx matmul: %s, %s, %s" % (type(a), type(b), type(c)))
    print("xilinx matmul: %s, %s, %s" % (a.dtype, b.dtype, c.dtype))
    print("Call count", xilinx_matmul_pynq.call_cnt)

    # import pdb;pdb.set_trace()

    # (m, k) = a.asnumpy().shape
    # (k, n) = b.asnumpy().shape
    m, k = get_const_tuple(a.shape)
    n, k = get_const_tuple(b.shape)
    #import pdb;pdb.set_trace()
    print("xilinx matmul: a shape", m, k)
    print("xilinx matmul: b shape", k, n)
    # cm, cn = c.asnumpy().shape
    cm, cn = get_const_tuple(c.shape)
    print("xilinx matmul: > c shape", cm, cn)
    # assert (m, n) == (cm, cn)
    if (m, n) != (cm, cn):
        print("xilinx matmul: c shape differs", (cm, cn), (m, n))
        # import pdb;pdb.set_trace()
    print("xilinx matmul: >> c shape", m, n)

    # import pdb;pdb.set_trace()
    Nbanks=8
    Nmat=3
    Tsize=1024
    Nvec=14

    k_pad = (Tsize - k) if (Tsize - k) > 0 else 0
    m_pad = (Nvec - m) if (Nvec - m) > 0 else 0
    print("m_pad, k_pad", m_pad, k_pad)
    v = np.pad(a.asnumpy(), [(0,m_pad), (0, k_pad)], mode='constant')

    # b: w, pad
    n_pad = (Nmat*Tsize)-n if (Nmat*Tsize)-n > 0 else 0
    print("n_pad", n_pad)
    w = np.pad(b.asnumpy(), [(0,n_pad),(0, k_pad)], mode='constant')


    if not pynq or n > (Nmat*Tsize) or k > Tsize:
        # a: v, pad

        outbuf = np.matmul(w, v.T).T
        outbuf1 = outbuf[:,:cn]
        tvm.nd.array(outbuf1.astype(c.dtype)).copyto(c)

        # c_np = np.matmul(a.asnumpy(), b.asnumpy())
        # print(c_np.shape)
        # import pdb;pdb.set_trace()

        # tvm.nd.array(np.matmul(a.asnumpy(), b.asnumpy().T).astype(c.dtype)).copyto(c)

        cm, cn = get_const_tuple(c.shape)

        print("numpy >>> xilinx matmul: >>> c shape", cm, cn)
    else:
        # TODO add pynq calls
        print("test")

        # data prep
        source_w_split_np = []
        for i in range(Nbanks):
            source_w_split_np.append(np.zeros((Nmat*Tsize,Tsize//Nbanks)))
        # a : w
        source_w_split_np = np.hsplit(w, Nbanks)
        print('w.shape', source_w_split_np[0].shape)

        # allocation buffer on the pynq

        # vector
        source_v = pynq.allocate(shape=(Nvec,Tsize), dtype=np.int16, target=xilinx_matmul_pynq.ol.HBM14)
        # a : v
        source_v[:] = v

        # W
        source_w = [
            pynq.allocate(shape=(Nmat*Tsize,Tsize//Nbanks), dtype=np.int16, target=xilinx_matmul_pynq.ol.HBM0),
            pynq.allocate(shape=(Nmat*Tsize,Tsize//Nbanks), dtype=np.int16, target=xilinx_matmul_pynq.ol.HBM4),
            pynq.allocate(shape=(Nmat*Tsize,Tsize//Nbanks), dtype=np.int16, target=xilinx_matmul_pynq.ol.HBM8),
            pynq.allocate(shape=(Nmat*Tsize,Tsize//Nbanks), dtype=np.int16, target=xilinx_matmul_pynq.ol.HBM12),
            pynq.allocate(shape=(Nmat*Tsize,Tsize//Nbanks), dtype=np.int16, target=xilinx_matmul_pynq.ol.HBM16),
            pynq.allocate(shape=(Nmat*Tsize,Tsize//Nbanks), dtype=np.int16, target=xilinx_matmul_pynq.ol.HBM20),
            pynq.allocate(shape=(Nmat*Tsize,Tsize//Nbanks), dtype=np.int16, target=xilinx_matmul_pynq.ol.HBM24),
            pynq.allocate(shape=(Nmat*Tsize,Tsize//Nbanks), dtype=np.int16, target=xilinx_matmul_pynq.ol.HBM26)]

        for i in range(len(source_w)):
            #     source_w[i][:] = np.random.randint(-2^15, high=2^15-1, dtype=np.int16, size=(Nmat*Tsize//Nbanks,Tsize))
            source_w[i][:] = source_w_split_np[i]

        # output
        outbuf = pynq.allocate((Tsize*Nmat,Nvec), dtype=np.int32, target=xilinx_matmul_pynq.ol.HBM14)
        print('outbuf.shape', outbuf.shape)


        # move the data to allocated buf

        source_v.sync_to_device()
        for i in range(8):
            source_w[i].sync_to_device()

        # call the kernel to do matmul
        xilinx_matmul_pynq.ol.feeder_1.call(
            source_v,
            source_w[0],
            source_w[1],
            source_w[2],
            source_w[3],
            source_w[4],
            source_w[5],
            source_w[6],
            source_w[7],
            outbuf, 0)

        # move the data back
        outbuf.sync_from_device()

        print("------ pynq inputs -------")
        print("w", source_w[0], source_w[0].shape)
        print("v", source_v, source_v.shape)
        print("------ pynq outputbuf -------")

        outbuf1 = outbuf.T
        outbuf2 = outbuf1[:cm,:cn]
        tvm.nd.array(outbuf2.astype(c.dtype)).copyto(c)

        print(outbuf2, outbuf2.dtype, outbuf2.shape)

        print(c, c.dtype, c.shape)
        print("------ pynq done ------------")
    end = time.time()
    print(end - start)


@autotvm.register_topi_compute("dense_nopack.xilinx_fpga")
def dense_nopack(cfg, data, weight, bias=None, out_dtype=None):
    """Compute dense without packing"""
    print("bias", bias)
    print("data_dtype", data.dtype)
    print("weight_dtype", weight.dtype)
    print("out_dtype", out_dtype)

    if out_dtype is None:
        out_dtype = data.dtype


    M, K = get_const_tuple(data.shape)
    N, _ = get_const_tuple(weight.shape)
    print("data", M, K)
    print("weight", N, _)
    print("bias", bias)

    # create tuning space
    # cfg.define_split("tile_y", 32 if isinstance(M, tvm.tir.Var) else M, num_outputs=2)
    # cfg.define_split("tile_x", 32 if isinstance(N, tvm.tir.Var) else N, num_outputs=2)
    # cfg.define_split("tile_k", 32 if isinstance(K, tvm.tir.Var) else K, num_outputs=2)
    # if cfg.is_fallback:
    #     _default_dense_nopack_config(cfg, M, N, K)
    #
    # vec = cfg["tile_k"].size[-1]
    # k = te.reduce_axis((0, K // vec), "k")
    # CC = te.compute(
    #     (M, N, vec),
    #     lambda z, y, x: te.sum(
    #         data[z, k * vec + x].astype(out_dtype) * weight[y, k * vec + x].astype(out_dtype),
    #         axis=k,
    #         ),
    # )
    #
    # kk = te.reduce_axis((0, vec), "kk")
    # C = te.compute((M, N), lambda y, x: te.sum(CC[y, x, kk], axis=kk), tag="dense_nopack")
    # if bias is not None:
    #     C = te.compute((M, N), lambda i, j: C[i, j] + bias[j].astype(out_dtype), tag=tag.BROADCAST)

    out = te.placeholder((M,N,), name="out", dtype=out_dtype)
    CC = te.extern(
        (M, N),
        [data, weight],
        lambda ins, outs: tvm.tir.call_packed("tvm.contrib.xilinx_matmul_pynq", ins[0], ins[1], outs[0]),
        dtype=out_dtype,
        name="matmul_pynq",
    )

    # kk = te.reduce_axis((0, vec), "kk")
    # C = te.compute((M, N), lambda y, x: te.sum(CC[y, x, kk], axis=kk), tag="dense_nopack")
    if bias is not None:
        C = te.compute((M, N), lambda i, j: CC[i, j] + bias[j].astype(out_dtype))
        return C

    return CC

"""
Test results
"""
"""
m, n, k = 2, 3, 4
A = te.placeholder((m,k), name="A")
B = te.placeholder((k,n), name="B")
C = te.extern(
    (m, n),
    [A, B],
    lambda ins, outs: tvm.tir.call_packed("tvm.contrib.xilinx_matmul_pynq", ins[0], ins[1], outs[0]),
    name="C",
)

s = te.create_schedule(C.op)
print(tvm.lower(s, [A, B, C],  simple_mode=True))

ctx = tvm.cpu(0)
f = tvm.build(s, [A, B, C], "llvm")
a = tvm.nd.array(np.random.uniform(size=(m,k)).astype(A.dtype), ctx)
b = tvm.nd.array(np.random.uniform(size=(k,n)).astype(B.dtype), ctx)
c = tvm.nd.array(np.random.uniform(size=(m,n)).astype(C.dtype), ctx)

f(a, b, c)
f(a, b, c)

tvm.testing.assert_allclose(np.matmul(a.asnumpy(), b.asnumpy()), c.asnumpy(), rtol=1e-5)

"""
