import numpy as np

def ex1_create_tensor():
    scalar = np.array(42)
    vector = np.array([1,2,3,4,5])
    matrix = np.zeros((2,3))
    tensor = np.ones((3,4,5))

    assert scalar.shape == ()
    assert vector.shape ==(5,)
    assert matrix.shape == (2,3)
    assert tensor.shape == (3,4,5)
    print("✅ ex1 passed")


def ex2_operations():
    a = np.array([1,2,3,4])
    b = np.array([10,20,30,40])

    add = a + b
    mul = a * b
    dot = np.dot(a,b)

    assert  np.array_equal(add,[11,22,33,44])
    assert np.array_equal(mul,[10,40,90,160])
    assert dot == 300
    print('✅ ex2 passed')

def ex3_broadcasting():
    x = np.ones((3,4))
    y = np.array([1,2,3,4])
    result = x + y

    assert result.shape == (3,4)
    assert np.array_equal(result[0],[2,3,4,5])
    print("✅ ex3 passed")

def ex4_matmul_manual():
    A = np.array([[1,2],[3,4]])
    B = np.array([[5,6],[7,8]])

    result = np.zeros((A.shape[0],B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                result[i,j] += A[i,k] * B[k,j]

    assert np.array_equal(result,[[19,22],[43,50]])
    print('✅ ex4 passed')


def ex5_reshape_transpose():
    x = np.arange(12)

    reshaped = x.reshape(3,4)
    transposed = reshaped.T
    flat = transposed.flatten()

    assert reshaped.shape == (3,4)
    assert transposed.shape == (4,3)
    assert flat.shape == (12,)
    print("✅ ex5 passed")

def ex6_normalization():
    x = np.array([[1.,2.,3.],
                  [4.,5.,6.],
                  [7.,8.,9.]])
    normalized = (x - x.mean())/x.std()

    assert np.isclose(normalized.mean(), 0, atol=1e-7)
    assert np.isclose(normalized.std(), 1, atol=1e-7)
    print("✅ ex6 passed")

def ex7_one_hot():
    label = np.array([0,2,1,3])
    num = 4

    one_hot = np.zeros((len(label),num))
    one_hot[np.arange(len(label)),label] = 1

    expected = np.array([[1,0,0,0],
                         [0,0,1,0],
                         [0,1,0,0],
                         [0,0,0,1]])
    assert np.array_equal(one_hot,expected)
    print("✅ ex7 passed")

if __name__ == "__main__":
    ex1_create_tensor()
    ex2_operations()
    ex3_broadcasting()
    ex4_matmul_manual()
    ex5_reshape_transpose()
    ex6_normalization()
    ex7_one_hot()
    print("\n🎉 All Day 1 exercises complete!")