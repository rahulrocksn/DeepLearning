#
import argparse
import numpy as onp
import numpy.typing as onpt
import multiprocessing as mp
import torch
from typing import Callable, List
from functools import partial


def rotate(
    mat: onpt.NDArray[onp.generic], d: int,
    /,
) -> onpt.NDArray[onp.generic]:
    R"""
    Rotate a 2D matrix.
    """
    #
    return onp.rot90(mat, d)


def flip(
    mat: onpt.NDArray[onp.generic], d: int,
    /,
) -> onpt.NDArray[onp.generic]:
    R"""
    Flip a 2D matrix.
    """
    #
    if d == 0:
        #
        return mat
    elif d == 1:
        return onp.flip(mat, 1)
    elif d == 2:
        return onp.flip(mat, 0)
    elif d == 3:
        return onp.flip(mat, (0, 1))
    else:
        #
        raise RuntimeError("Unsupported flipping argument.")


def averaged(mat: onpt.NDArray[onp.generic], /) -> onpt.NDArray[onp.generic]:
    R"""
    Average of all transformations.
    """
    #
    buf = []
    for d_rotate in range(4):
        #
        for d_flip in range(4):
            #
            buf.append(flip(rotate(mat, d_rotate), d_flip))
    mat = onp.stack(buf)
    mat = onp.mean(mat, axis=0)
    return mat


def transformed(
    i: int,
    /,
    *,
    f: Callable[[onpt.NDArray[onp.generic]], onpt.NDArray[onp.generic]],
    shape: List[int],
) -> onpt.NDArray[onp.generic]:
    R"""
    Get transformed one-hot vector.
    """
    #
    n = onp.prod(shape)
    onehot = onp.zeros(n)
    onehot[i] = 1
    onehot = f(onp.reshape(onehot, tuple(shape)))
    return onp.reshape(onehot, (n,))


def subspace(size: int, /, *, n_cpus: int):
    R"""
    Get invariant subspace.
    """
    #
    transforms = []
    for d_rotate in range(4):
        #
        for d_flip in range(4):
            #
            transform = lambda mat: flip(rotate(mat, d_rotate), d_flip)
            transforms.append(transform)
    n = size * size

    #
    if n_cpus > 0:
        #
        pool = mp.Pool(n_cpus)
        buf = (
            pool.map(
                partial(transformed, f=averaged, shape=(size, size)), range(n),
            )
        )
        pool.close()
        pool.join()
    else:
        #
        buf = (
            [transformed(i, f=averaged, shape=[size, size]) for i in range(n)]
        )
    transmatrix = onp.stack(buf, axis=1)

    #
    (_, eigenvalues, eigenvectors) = (
        onp.linalg.svd(transmatrix, hermitian=True)
    )
    rank = onp.linalg.matrix_rank(onp.diag(eigenvalues), hermitian=True)
    eigenvectors = eigenvectors[:rank]

    #
    with open("rf-{:d}.npy".format(size), "wb") as file:
        #
        onp.save(file, transmatrix)
        onp.save(file, eigenvectors)


def main(*ARGS) -> None:
    R"""
    Main.
    """
    #
    parser = argparse.ArgumentParser(description="Structure Execution")
    parser.add_argument(
        "--size",
        type=int, required=True,
        help="G-invariant squared image size (height or width).",
    )
    parser.add_argument(
        "--num-cpus",
        type=int, required=False, default=0, help="Number of cpus to be used.",
    )
    args = parser.parse_args() if len(ARGS) == 0 else parser.parse_args(ARGS)

    #
    size = args.size
    n_cpus = args.num_cpus

    #
    subspace(size, n_cpus=n_cpus)


#
if __name__ == "__main__":
    #
    main()