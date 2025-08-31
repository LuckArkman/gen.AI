using System;
using System.Linq; // Necessário para SequenceEqual

namespace Core;

public static class TensorExtensions
{
    public static Tensor MatMul(this Tensor A, Tensor B)
    {
        // Adaptação para multiplicação de tensores 1D e 2D
        // A é [M, K] ou [K], B é [K, N] ou [K]
        Tensor A_expanded = (A.shape.Length == 1) ? new Tensor(A.GetData(), new int[] { 1, A.shape[0] }) : A;
        Tensor B_expanded = (B.shape.Length == 1) ? new Tensor(B.GetData(), new int[] { B.shape[0], 1 }) : B;

        if (A_expanded.shape[1] != B_expanded.shape[0])
            throw new ArgumentException($"Matrix dimensions mismatch for multiplication. A.cols ({A_expanded.shape[1]}) != B.rows ({B_expanded.shape[0]}).");

        int M = A_expanded.shape[0];
        int K = A_expanded.shape[1];
        int N = B_expanded.shape[1];

        double[] resultData = new double[M * N];
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                double sum = 0;
                for (int k = 0; k < K; k++)
                {
                    sum += A_expanded.Infer(new int[] { i, k }) * B_expanded.Infer(new int[] { k, j });
                }
                resultData[i * N + j] = sum;
            }
        }
        // Se o resultado é uma matriz 1xN, retorna um Tensor 1D de tamanho N
        if (M == 1 && A.shape.Length == 1) // Se A era originalmente 1D e o resultado é uma linha
        {
            return new Tensor(resultData, new int[] { N });
        }
        if (M == 1 && N == 1) // Se o resultado é um escalar (e.g. produto interno)
        {
            return new Tensor(resultData, new int[] { 1 });
        }
        return new Tensor(resultData, new int[] { M, N });
    }

    public static Tensor Add(this Tensor A, Tensor B)
    {
        // Adição simples elemento a elemento (formas idênticas)
        if (A.shape.SequenceEqual(B.shape))
        {
            double[] resultData = new double[A.GetTotalSize()];
            double[] aData = A.GetData();
            double[] bData = B.GetData();
            for (int i = 0; i < A.GetTotalSize(); i++)
            {
                resultData[i] = aData[i] + bData[i];
            }
            return new Tensor(resultData, A.shape);
        }
        // Broadcasting de vetor (bias) para matriz
        if (A.shape.Length == 2 && B.shape.Length == 1 && A.shape[1] == B.shape[0])
        {
            int rows = A.shape[0];
            int cols = A.shape[1];
            double[] resultData = new double[rows * cols];
            double[] aData = A.GetData();
            double[] bData = B.GetData(); // Bias
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    resultData[r * cols + c] = aData[r * cols + c] + bData[c];
                }
            }
            return new Tensor(resultData, A.shape);
        }
        // Broadcasting de escalar para vetor/matriz (se um dos tensores for de tamanho 1)
        if (A.GetTotalSize() == 1)
        {
            double scalar = A.GetData()[0];
            double[] resultData = new double[B.GetTotalSize()];
            double[] bData = B.GetData();
            for (int i = 0; i < B.GetTotalSize(); i++)
            {
                resultData[i] = scalar + bData[i];
            }
            return new Tensor(resultData, B.shape);
        }
        if (B.GetTotalSize() == 1)
        {
            double scalar = B.GetData()[0];
            double[] resultData = new double[A.GetTotalSize()];
            double[] aData = A.GetData();
            for (int i = 0; i < A.GetTotalSize(); i++)
            {
                resultData[i] = aData[i] + scalar;
            }
            return new Tensor(resultData, A.shape);
        }

        throw new ArgumentException($"Tensors must have same shape for addition or be compatible for broadcasting. A:{string.Join("x",A.shape)}, B:{string.Join("x",B.shape)}");
    }

    public static Tensor Apply(this Tensor A, Func<double, double> activationFunction)
    {
        double[] resultData = new double[A.GetTotalSize()];
        double[] aData = A.GetData();
        for (int i = 0; i < A.GetTotalSize(); i++)
        {
            resultData[i] = activationFunction(aData[i]);
        }
        return new Tensor(resultData, A.shape);
    }
    
    public static Tensor ElementWiseMultiply(this Tensor A, Tensor B)
    {
        if (!A.shape.SequenceEqual(B.shape))
            throw new ArgumentException($"Tensors must have same shape for element-wise multiplication. A:{string.Join("x",A.shape)}, B:{string.Join("x",B.shape)}");

        double[] resultData = new double[A.GetTotalSize()];
        double[] aData = A.GetData();
        double[] bData = B.GetData();
        for (int i = 0; i < A.GetTotalSize(); i++)
        {
            resultData[i] = aData[i] * bData[i];
        }
        return new Tensor(resultData, A.shape);
    }
}