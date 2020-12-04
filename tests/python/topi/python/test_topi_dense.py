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
"""Test code for dense operator"""
import numpy as np
import tvm
from tvm import te
from tvm import topi
import tvm.topi.testing
from tvm.topi.utils import get_const_tuple
from tvm.contrib.pickle_memoize import memoize

from common import Int8Fallback
import tvm.testing

_dense_implement = {
    # "generic": [(topi.nn.dense, topi.generic.schedule_dense)],
    "cpu": [
        (topi.xilinx_fpga.dense_nopack, topi.generic.schedule_extern),
        # (topi.x86.dense_nopack, topi.x86.schedule_dense_nopack),
        # (topi.x86.dense_pack, topi.x86.schedule_dense_pack),
    ],
    # "gpu": [
    #     (topi.cuda.dense_small_batch, topi.cuda.schedule_dense_small_batch),
    #     (topi.cuda.dense_large_batch, topi.cuda.schedule_dense_large_batch),
    # ],
    # "mali": [(topi.mali.dense, topi.mali.schedule_dense)],
    # "bifrost": [(topi.bifrost.dense, topi.bifrost.schedule_dense)],
    # "rocm": [(topi.rocm.dense, topi.rocm.schedule_dense)],
    # "hls": [(topi.nn.dense, topi.hls.schedule_dense)],
}


def verify_dense(batch, in_dim, out_dim, use_bias=True):
    # dtype = "float32"
    # out_dtype = "float32"
    #dtype = "int8"
    dtype = "int16"
    out_dtype = "int32"
    A = te.placeholder((batch, in_dim), name="A", dtype=dtype)
    B = te.placeholder((out_dim, in_dim), name="B", dtype=dtype)
    C = te.placeholder((out_dim,), name="C", dtype=out_dtype)
    dtype = A.dtype

    # use memoize to pickle the test data for next time use
    @memoize("topi.tests.test_topi_dense")
    def get_ref_data():
        # W
        low = -1000
        high = 1000
        a_np = np.random.randint(low, high, dtype=dtype, size=(batch, in_dim)) #.astype(dtype)
        # V
        low = 1 # -2^15
        high =  200# 2^15-1
        b_np = np.random.randint(low, high, dtype=dtype, size=(out_dim, in_dim)) #.astype(dtype)

        c_np = np.random.randint(-128, 128, size=(out_dim,)).astype(out_dtype)
        # skip relu
        """
        if use_bias:
            d_np = np.maximum(np.matmul(a_np, b_np.T, dtype=out_dtype) + c_np, 0.0)
        else:
            d_np = np.maximum(np.matmul(a_np, b_np.T, dtype=out_dtype), 0.0)
        """
        if use_bias:
            d_np = np.matmul(a_np, b_np.T, dtype=out_dtype) + c_np
        else:
            d_np = np.matmul(a_np, b_np.T, dtype=out_dtype)

        return (a_np, b_np, c_np, d_np)

    # get the test data
    a_np, b_np, c_np, d_np = get_ref_data()

    def check_device(device, ctx):
        print("Running on target: %s" % device)
        for fcompute, fschedule in tvm.topi.testing.dispatch(device, _dense_implement):
            with tvm.target.Target(device):
                print("A dtype", A.dtype)
                print("B dtype", B.dtype)
                print("C dtype", C.dtype)
                D = fcompute(A, B, C if use_bias else None, out_dtype)
                # skip relu
                """
                print("D dtype", D.dtype)
                # D = fcompute(A, B, C, C.dtype)
                D = topi.nn.relu(D)
                print(">D dtype", D.dtype)
                """
                s = fschedule([D])
            a = tvm.nd.array(a_np, ctx)
            b = tvm.nd.array(b_np, ctx)
            c = tvm.nd.array(c_np, ctx)
            d = tvm.nd.array(np.zeros(get_const_tuple(D.shape), dtype=out_dtype), ctx)
            f = tvm.build(s, [A, B, C, D], device, name="dense")
            f(a, b, c, d)
            # import pdb;pdb.set_trace()
            print("a, w", a, a.dtype)
            print("b, v", b, b.dtype)
            print("c_np", c_np)
            print("d, outbuf", d, d.dtype)
            print("d_np, outbuf_np", d_np, d_np.dtype)
            tvm.testing.assert_allclose(d.asnumpy(), d_np, rtol=1e-5)

    for device, ctx in tvm.testing.enabled_targets():
        check_device(device, ctx)


def verify_dense_int8(batch, in_dim, out_dim, use_bias=True):
    dtype = "int8"
    out_dtype = "int32"
    A = te.placeholder((batch, in_dim), name="A", dtype=dtype)
    B = te.placeholder((out_dim, in_dim), name="B", dtype=dtype)
    C = te.placeholder((out_dim,), name="C", dtype=out_dtype)

    # use memoize to pickle the test data for next time use
    @memoize("topi.tests.test_topi_dense_int8")
    def get_ref_data():
        a_np = np.random.randint(low=-128, high=127, size=(batch, in_dim)).astype(dtype)
        b_np = np.random.randint(low=-128, high=127, size=(out_dim, in_dim)).astype(dtype)
        c_np = np.random.randint(low=-128, high=127, size=(out_dim,)).astype(out_dtype)
        d_np = np.dot(a_np.astype(out_dtype), b_np.T.astype(out_dtype))
        if use_bias:
            d_np += c_np
        d_np = np.maximum(d_np, 0.0)
        return (a_np, b_np, c_np, d_np)

    # get the test data
    a_np, b_np, c_np, d_np = get_ref_data()

    def check_device(device):
        if False:
            ctx = tvm.context(device, 0)
            if device == "cuda" and not tvm.contrib.nvcc.have_int8(ctx.compute_version):
                print("Skip because int8 intrinsics are not available")
                return

            print("Running on target: %s" % device)
            with tvm.target.Target(device):
                D = topi.cuda.dense_int8(A, B, C if use_bias else None, out_dtype)
                D = topi.nn.relu(D)
                s = topi.cuda.schedule_dense_int8([D])
            a = tvm.nd.array(a_np, ctx)
            b = tvm.nd.array(b_np, ctx)
            c = tvm.nd.array(c_np, ctx)
            d = tvm.nd.array(np.zeros(get_const_tuple(D.shape), dtype=out_dtype), ctx)
            f = tvm.build(s, [A, B, C, D], device, name="dense")
            f(a, b, c, d)
            tvm.testing.assert_allclose(d.asnumpy(), d_np, rtol=1e-5)
        else:
            print("Running on target: %s" % device)
            for fcompute, fschedule in tvm.topi.testing.dispatch(device, _dense_implement):
                with tvm.target.Target(device):
                    D = fcompute(A, B, C if use_bias else None)
                    D = topi.nn.relu(D)
                    s = fschedule([D])
                a = tvm.nd.array(a_np, ctx)
                b = tvm.nd.array(b_np, ctx)
                c = tvm.nd.array(c_np, ctx)
                d = tvm.nd.array(np.zeros(get_const_tuple(D.shape), dtype=dtype), ctx)
                f = tvm.build(s, [A, B, C, D], device, name="dense")
                f(a, b, c, d)
                tvm.testing.assert_allclose(d.asnumpy(), d_np, rtol=1e-5)


    for device in ["cuda"]:
        check_device(device)


@tvm.testing.uses_gpu
def test_dense():
    verify_dense(3072, 1024, 14, use_bias=True)
    verify_dense(3072, 1024, 14, use_bias=False)
    #verify_dense(1, 1024, 1000, use_bias=True)
    #verify_dense(1, 1024, 1000, use_bias=False)
    #verify_dense(2, 1024, 1000, use_bias=True)
    #verify_dense(128, 1024, 1000, use_bias=False)
    #verify_dense(128, 1024, 1000, use_bias=True)


@tvm.testing.requires_cuda
@tvm.testing.requires_gpu
def test_dense_int8():
    with Int8Fallback():
        verify_dense_int8(2, 1024, 1000, use_bias=True)
        verify_dense_int8(2, 1024, 1000, use_bias=False)


if __name__ == "__main__":
    test_dense()
    # test_dense_int8()
