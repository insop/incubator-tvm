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
Add operation for xilinx fpga
=========================
**Author**: `Insop Song <https://insop.github.io>`_

This operator is for calling xilinx fpga kernel for addition.
This is mainly for initial testing.

"""

import tvm
from tvm import te
import tvm.testing
import numpy as np


@tvm.register_func("tvm.contrib.xilinx_add_pynq")
def xilinx_add_pynq(a, b, c):

    debug = True

    # use call_cnt for lazy initilization
    if not hasattr(xilinx_add_pynq, 'call_cnt'):
        xilinx_add_pynq.call_cnt = 0

    # laze initialization, such as xclbin loading for fpga
    if xilinx_add_pynq.call_cnt == 1:
        print("xclbin load")

    xilinx_add_pynq.call_cnt += 1
    print("xilinx add: %s, %s, %s" % (type(a), type(b), type(c)))
    print("Call count", xilinx_add_pynq.call_cnt)

    (m, ) = a.asnumpy().shape
    (m, ) = b.asnumpy().shape
    #import pdb;pdb.set_trace()
    print("xilinx add: a shape", m)
    print("xilinx add: b shape", m)
    cm, = c.asnumpy().shape
    assert (m,) == (cm,)
    print("xilinx matmul: c shape", m)

    if debug:
        tvm.nd.array(np.add(a.asnumpy(), b.asnumpy())).copyto(c)
    else:
        # TODO add pynq calls
        c_zeros = np.zeros_like(c.asnumpy())
        tvm.nd.array(c_zeros).copyto(c)



"""
Test results
"""
"""
m = 16
A = te.placeholder((m,), name="A")
B = te.placeholder((m,), name="B")
C = te.extern(
    (m,),
    [A, B],
    lambda ins, outs: tvm.tir.call_packed("tvm.contrib.xilinx_add_pynq", ins[0], ins[1], outs[0]),
    name="C",
)

s = te.create_schedule(C.op)
print(tvm.lower(s, [A, B, C],  simple_mode=True))

ctx = tvm.cpu(0)
f = tvm.build(s, [A, B, C], "llvm")
a = tvm.nd.array(np.random.uniform(size=(m,)).astype(A.dtype), ctx)
b = tvm.nd.array(np.random.uniform(size=(m,)).astype(B.dtype), ctx)
c = tvm.nd.array(np.random.uniform(size=(m,)).astype(C.dtype), ctx)

f(a, b, c)
f(a, b, c)

tvm.testing.assert_allclose(np.add(a.asnumpy(), b.asnumpy()), c.asnumpy(), rtol=1e-5)
"""

