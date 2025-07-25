using Cloo;
using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Core
{
    public class NeuralNetwork : IDisposable
    {
        private Tensor W_i, U_i, b_i, W_f, U_f, b_f, W_c, U_c, b_c, W_o, U_o, b_o, W_out, b_out;
        private readonly int inputSize, hiddenSize, outputSize, contextWindowSize;

        // Campos OpenCL
        private ComputeContext _context;
        private ComputeDevice _device;
        private ComputeCommandQueue _queue;
        private ComputeProgram _program;
        private ComputeKernel _matmulKernel;
        private ComputeKernel _sigmoidKernel;
        private ComputeKernel _tanhKernel;
        private ComputeKernel _softmaxKernel;
        private ComputeKernel _elementwiseAddKernel;
        private ComputeKernel _elementwiseMultiplyKernel;

        public int InputSize => inputSize;
        public int HiddenSize => hiddenSize;
        public int OutputSize => outputSize;

        public NeuralNetwork(int inputSize, int hiddenSize, int outputSize, int contextWindowSize)
        {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;
            this.contextWindowSize = contextWindowSize;

            Random rand = new Random();
            double sqrtFanIn = Math.Sqrt(2.0 / (inputSize + hiddenSize));
            W_i = new Tensor(InitializeWeights(inputSize, hiddenSize, sqrtFanIn, rand), new int[] { inputSize, hiddenSize });
            U_i = new Tensor(InitializeWeights(hiddenSize, hiddenSize, sqrtFanIn, rand), new int[] { hiddenSize, hiddenSize });
            b_i = new Tensor(new double[hiddenSize], new int[] { hiddenSize });
            W_f = new Tensor(InitializeWeights(inputSize, hiddenSize, sqrtFanIn, rand), new int[] { inputSize, hiddenSize });
            U_f = new Tensor(InitializeWeights(hiddenSize, hiddenSize, sqrtFanIn, rand), new int[] { hiddenSize, hiddenSize });
            b_f = new Tensor(new double[hiddenSize], new int[] { hiddenSize });
            W_c = new Tensor(InitializeWeights(inputSize, hiddenSize, sqrtFanIn, rand), new int[] { inputSize, hiddenSize });
            U_c = new Tensor(InitializeWeights(hiddenSize, hiddenSize, sqrtFanIn, rand), new int[] { hiddenSize, hiddenSize });
            b_c = new Tensor(new double[hiddenSize], new int[] { hiddenSize });
            W_o = new Tensor(InitializeWeights(inputSize, hiddenSize, sqrtFanIn, rand), new int[] { inputSize, hiddenSize });
            U_o = new Tensor(InitializeWeights(hiddenSize, hiddenSize, sqrtFanIn, rand), new int[] { hiddenSize, hiddenSize });
            b_o = new Tensor(new double[hiddenSize], new int[] { hiddenSize });
            W_out = new Tensor(InitializeWeights(hiddenSize, outputSize, Math.Sqrt(2.0 / hiddenSize), rand), new int[] { hiddenSize, outputSize });
            b_out = new Tensor(new double[outputSize], new int[] { outputSize });

            InitializeOpenCL();
        }

        private NeuralNetwork(int inputSize, int hiddenSize, int outputSize, int contextWindowSize,
            Tensor W_i, Tensor U_i, Tensor b_i,
            Tensor W_f, Tensor U_f, Tensor b_f,
            Tensor W_c, Tensor U_c, Tensor b_c,
            Tensor W_o, Tensor U_o, Tensor b_o,
            Tensor W_out, Tensor b_out)
        {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;
            this.contextWindowSize = contextWindowSize;
            this.W_i = W_i;
            this.U_i = U_i;
            this.b_i = b_i;
            this.W_f = W_f;
            this.U_f = U_f;
            this.b_f = b_f;
            this.W_c = W_c;
            this.U_c = U_c;
            this.b_c = b_c;
            this.W_o = W_o;
            this.U_o = U_o;
            this.b_o = b_o;
            this.W_out = W_out;
            this.b_out = b_out;

            InitializeOpenCL();
        }

        private void InitializeOpenCL()
        {
            try
            {
                var platform = ComputePlatform.Platforms[0];
                _context = new ComputeContext(ComputeDeviceTypes.Gpu, new ComputeContextPropertyList(platform), null, IntPtr.Zero);
                _device = _context.Devices[0];
                Console.WriteLine($"Dispositivo OpenCL selecionado: {_device.Name} (Tipo: {_device.Type})");
                _queue = new ComputeCommandQueue(_context, _device, ComputeCommandQueueFlags.None);

                string kernelPath = Path.Combine(AppContext.BaseDirectory, "/home/mplopes/Documentos/GitHub/gen.AI/generative/generative/Kernels/MatrixOperations.cl");
                if (!File.Exists(kernelPath))
                {
                    throw new FileNotFoundException($"Arquivo de kernel OpenCL não encontrado: {kernelPath}");
                }
                string kernelSource = File.ReadAllText(kernelPath);
                _program = new ComputeProgram(_context, kernelSource);
                _program.Build(null, null, null, IntPtr.Zero);

                _matmulKernel = _program.CreateKernel("matmul_forward");
                _sigmoidKernel = _program.CreateKernel("sigmoid_forward");
                _tanhKernel = _program.CreateKernel("tanh_forward");
                _softmaxKernel = _program.CreateKernel("softmax_forward");
                _elementwiseAddKernel = _program.CreateKernel("elementwise_add_forward");
                _elementwiseMultiplyKernel = _program.CreateKernel("elementwise_multiply");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erro ao inicializar OpenCL: {ex.Message}");
                throw;
            }
        }

        public void Dispose()
        {
            _matmulKernel?.Dispose();
            _sigmoidKernel?.Dispose();
            _tanhKernel?.Dispose();
            _softmaxKernel?.Dispose();
            _elementwiseAddKernel?.Dispose();
            _elementwiseMultiplyKernel?.Dispose();
            _program?.Dispose();
            _queue?.Dispose();
            _context?.Dispose();
        }

        private static double[] InitializeWeights(int rows, int cols, double scale, Random rand)
        {
            double[] weights = new double[rows * cols];
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = (rand.NextDouble() * 2 - 1) * scale;
            }
            return weights;
        }

        private double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
        private double Tanh(double x) => Math.Tanh(x);
        private double SigmoidDerivative(double sigmoidOutput) => sigmoidOutput * (1 - sigmoidOutput);
        private double TanhDerivative(double tanhOutput) => 1 - tanhOutput * tanhOutput;

        private (Tensor h_t, Tensor c_t, Tensor i_t, Tensor f_t, Tensor c_tilde, Tensor o_t) LSTMStep(Tensor x_t, Tensor h_prev, Tensor c_prev)
        {
            float[] x_t_data = x_t.GetData().Select(d => (float)d).ToArray();
            float[] h_prev_data = h_prev.GetData().Select(d => (float)d).ToArray();
            float[] c_prev_data = c_prev.GetData().Select(d => (float)d).ToArray();
            ComputeBuffer<float> xBuffer = new ComputeBuffer<float>(_context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, x_t_data);
            ComputeBuffer<float> hPrevBuffer = new ComputeBuffer<float>(_context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, h_prev_data);
            ComputeBuffer<float> cPrevBuffer = new ComputeBuffer<float>(_context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, c_prev_data);

            ComputeBuffer<float> wiBuffer = new ComputeBuffer<float>(_context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, W_i.GetData().Select(d => (float)d).ToArray());
            ComputeBuffer<float> uiBuffer = new ComputeBuffer<float>(_context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, U_i.GetData().Select(d => (float)d).ToArray());
            ComputeBuffer<float> biBuffer = new ComputeBuffer<float>(_context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, b_i.GetData().Select(d => (float)d).ToArray());
            ComputeBuffer<float> wfBuffer = new ComputeBuffer<float>(_context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, W_f.GetData().Select(d => (float)d).ToArray());
            ComputeBuffer<float> ufBuffer = new ComputeBuffer<float>(_context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, U_f.GetData().Select(d => (float)d).ToArray());
            ComputeBuffer<float> bfBuffer = new ComputeBuffer<float>(_context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, b_f.GetData().Select(d => (float)d).ToArray());
            ComputeBuffer<float> wcBuffer = new ComputeBuffer<float>(_context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, W_c.GetData().Select(d => (float)d).ToArray());
            ComputeBuffer<float> ucBuffer = new ComputeBuffer<float>(_context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, U_c.GetData().Select(d => (float)d).ToArray());
            ComputeBuffer<float> bcBuffer = new ComputeBuffer<float>(_context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, b_c.GetData().Select(d => (float)d).ToArray());
            ComputeBuffer<float> woBuffer = new ComputeBuffer<float>(_context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, W_o.GetData().Select(d => (float)d).ToArray());
            ComputeBuffer<float> uoBuffer = new ComputeBuffer<float>(_context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, U_o.GetData().Select(d => (float)d).ToArray());
            ComputeBuffer<float> boBuffer = new ComputeBuffer<float>(_context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, b_o.GetData().Select(d => (float)d).ToArray());

            float[] i_t_data = new float[hiddenSize];
            float[] f_t_data = new float[hiddenSize];
            float[] c_tilde_data = new float[hiddenSize];
            float[] o_t_data = new float[hiddenSize];
            float[] c_t_data = new float[hiddenSize];
            float[] h_t_data = new float[hiddenSize];
            ComputeBuffer<float> i_t_buffer = new ComputeBuffer<float>(_context, ComputeMemoryFlags.WriteOnly, hiddenSize);
            ComputeBuffer<float> f_t_buffer = new ComputeBuffer<float>(_context, ComputeMemoryFlags.WriteOnly, hiddenSize);
            ComputeBuffer<float> c_tilde_buffer = new ComputeBuffer<float>(_context, ComputeMemoryFlags.WriteOnly, hiddenSize);
            ComputeBuffer<float> o_t_buffer = new ComputeBuffer<float>(_context, ComputeMemoryFlags.WriteOnly, hiddenSize);
            ComputeBuffer<float> c_t_buffer = new ComputeBuffer<float>(_context, ComputeMemoryFlags.WriteOnly, hiddenSize);
            ComputeBuffer<float> h_t_buffer = new ComputeBuffer<float>(_context, ComputeMemoryFlags.WriteOnly, hiddenSize);
            ComputeBuffer<float> temp_buffer = new ComputeBuffer<float>(_context, ComputeMemoryFlags.WriteOnly, hiddenSize);

            // Porta de entrada: i_t = sigmoid(W_i * x_t + U_i * h_prev + b_i)
            _matmulKernel.SetMemoryArgument(0, xBuffer);
            _matmulKernel.SetMemoryArgument(1, wiBuffer);
            _matmulKernel.SetMemoryArgument(2, temp_buffer);
            _matmulKernel.SetValueArgument(3, 1);
            _matmulKernel.SetValueArgument(4, x_t.shape[0]);
            _matmulKernel.SetValueArgument(5, hiddenSize);
            _queue.Execute(_matmulKernel, null, new long[] { 1, hiddenSize }, null, null);

            _matmulKernel.SetMemoryArgument(0, hPrevBuffer);
            _matmulKernel.SetMemoryArgument(1, uiBuffer);
            _matmulKernel.SetMemoryArgument(2, i_t_buffer);
            _matmulKernel.SetValueArgument(3, 1);
            _matmulKernel.SetValueArgument(4, hiddenSize);
            _matmulKernel.SetValueArgument(5, hiddenSize);
            _queue.Execute(_matmulKernel, null, new long[] { 1, hiddenSize }, null, null);

            _elementwiseAddKernel.SetMemoryArgument(0, temp_buffer);
            _elementwiseAddKernel.SetMemoryArgument(1, i_t_buffer);
            _elementwiseAddKernel.SetMemoryArgument(2, i_t_buffer);
            _elementwiseAddKernel.SetValueArgument(3, hiddenSize);
            _queue.Execute(_elementwiseAddKernel, null, new long[] { hiddenSize }, null, null);

            _elementwiseAddKernel.SetMemoryArgument(0, i_t_buffer);
            _elementwiseAddKernel.SetMemoryArgument(1, biBuffer);
            _elementwiseAddKernel.SetMemoryArgument(2, i_t_buffer);
            _elementwiseAddKernel.SetValueArgument(3, hiddenSize);
            _queue.Execute(_elementwiseAddKernel, null, new long[] { hiddenSize }, null, null);

            _sigmoidKernel.SetMemoryArgument(0, i_t_buffer);
            _sigmoidKernel.SetMemoryArgument(1, i_t_buffer);
            _sigmoidKernel.SetValueArgument(2, hiddenSize);
            _queue.Execute(_sigmoidKernel, null, new long[] { hiddenSize }, null, null);

            GCHandle i_t_handle = GCHandle.Alloc(i_t_data, GCHandleType.Pinned);
            try
            {
                _queue.Read(i_t_buffer, true, 0, hiddenSize, i_t_handle.AddrOfPinnedObject(), null);
            }
            finally
            {
                i_t_handle.Free();
            }

            // Porta de esquecimento: f_t = sigmoid(W_f * x_t + U_f * h_prev + b_f)
            _matmulKernel.SetMemoryArgument(0, xBuffer);
            _matmulKernel.SetMemoryArgument(1, wfBuffer);
            _matmulKernel.SetMemoryArgument(2, temp_buffer);
            _matmulKernel.SetValueArgument(3, 1);
            _matmulKernel.SetValueArgument(4, x_t.shape[0]);
            _matmulKernel.SetValueArgument(5, hiddenSize);
            _queue.Execute(_matmulKernel, null, new long[] { 1, hiddenSize }, null, null);

            _matmulKernel.SetMemoryArgument(0, hPrevBuffer);
            _matmulKernel.SetMemoryArgument(1, ufBuffer);
            _matmulKernel.SetMemoryArgument(2, f_t_buffer);
            _matmulKernel.SetValueArgument(3, 1);
            _matmulKernel.SetValueArgument(4, hiddenSize);
            _matmulKernel.SetValueArgument(5, hiddenSize);
            _queue.Execute(_matmulKernel, null, new long[] { 1, hiddenSize }, null, null);

            _elementwiseAddKernel.SetMemoryArgument(0, temp_buffer);
            _elementwiseAddKernel.SetMemoryArgument(1, f_t_buffer);
            _elementwiseAddKernel.SetMemoryArgument(2, f_t_buffer);
            _elementwiseAddKernel.SetValueArgument(3, hiddenSize);
            _queue.Execute(_elementwiseAddKernel, null, new long[] { hiddenSize }, null, null);

            _elementwiseAddKernel.SetMemoryArgument(0, f_t_buffer);
            _elementwiseAddKernel.SetMemoryArgument(1, bfBuffer);
            _elementwiseAddKernel.SetMemoryArgument(2, f_t_buffer);
            _elementwiseAddKernel.SetValueArgument(3, hiddenSize);
            _queue.Execute(_elementwiseAddKernel, null, new long[] { hiddenSize }, null, null);

            _sigmoidKernel.SetMemoryArgument(0, f_t_buffer);
            _sigmoidKernel.SetMemoryArgument(1, f_t_buffer);
            _sigmoidKernel.SetValueArgument(2, hiddenSize);
            _queue.Execute(_sigmoidKernel, null, new long[] { hiddenSize }, null, null);

            GCHandle f_t_handle = GCHandle.Alloc(f_t_data, GCHandleType.Pinned);
            try
            {
                _queue.Read(f_t_buffer, true, 0, hiddenSize, f_t_handle.AddrOfPinnedObject(), null);
            }
            finally
            {
                f_t_handle.Free();
            }

            // Porta da célula: c_tilde = tanh(W_c * x_t + U_c * h_prev + b_c)
            _matmulKernel.SetMemoryArgument(0, xBuffer);
            _matmulKernel.SetMemoryArgument(1, wcBuffer);
            _matmulKernel.SetMemoryArgument(2, temp_buffer);
            _matmulKernel.SetValueArgument(3, 1);
            _matmulKernel.SetValueArgument(4, x_t.shape[0]);
            _matmulKernel.SetValueArgument(5, hiddenSize);
            _queue.Execute(_matmulKernel, null, new long[] { 1, hiddenSize }, null, null);

            _matmulKernel.SetMemoryArgument(0, hPrevBuffer);
            _matmulKernel.SetMemoryArgument(1, ucBuffer);
            _matmulKernel.SetMemoryArgument(2, c_tilde_buffer);
            _matmulKernel.SetValueArgument(3, 1);
            _matmulKernel.SetValueArgument(4, hiddenSize);
            _matmulKernel.SetValueArgument(5, hiddenSize);
            _queue.Execute(_matmulKernel, null, new long[] { 1, hiddenSize }, null, null);

            _elementwiseAddKernel.SetMemoryArgument(0, temp_buffer);
            _elementwiseAddKernel.SetMemoryArgument(1, c_tilde_buffer);
            _elementwiseAddKernel.SetMemoryArgument(2, c_tilde_buffer);
            _elementwiseAddKernel.SetValueArgument(3, hiddenSize);
            _queue.Execute(_elementwiseAddKernel, null, new long[] { hiddenSize }, null, null);

            _elementwiseAddKernel.SetMemoryArgument(0, c_tilde_buffer);
            _elementwiseAddKernel.SetMemoryArgument(1, bcBuffer);
            _elementwiseAddKernel.SetMemoryArgument(2, c_tilde_buffer);
            _elementwiseAddKernel.SetValueArgument(3, hiddenSize);
            _queue.Execute(_elementwiseAddKernel, null, new long[] { hiddenSize }, null, null);

            _tanhKernel.SetMemoryArgument(0, c_tilde_buffer);
            _tanhKernel.SetMemoryArgument(1, c_tilde_buffer);
            _tanhKernel.SetValueArgument(2, hiddenSize);
            _queue.Execute(_tanhKernel, null, new long[] { hiddenSize }, null, null);

            GCHandle c_tilde_handle = GCHandle.Alloc(c_tilde_data, GCHandleType.Pinned);
            try
            {
                _queue.Read(c_tilde_buffer, true, 0, hiddenSize, c_tilde_handle.AddrOfPinnedObject(), null);
            }
            finally
            {
                c_tilde_handle.Free();
            }

            // Porta de saída: o_t = sigmoid(W_o * x_t + U_o * h_prev + b_o)
            _matmulKernel.SetMemoryArgument(0, xBuffer);
            _matmulKernel.SetMemoryArgument(1, woBuffer);
            _matmulKernel.SetMemoryArgument(2, temp_buffer);
            _matmulKernel.SetValueArgument(3, 1);
            _matmulKernel.SetValueArgument(4, x_t.shape[0]);
            _matmulKernel.SetValueArgument(5, hiddenSize);
            _queue.Execute(_matmulKernel, null, new long[] { 1, hiddenSize }, null, null);

            _matmulKernel.SetMemoryArgument(0, hPrevBuffer);
            _matmulKernel.SetMemoryArgument(1, uoBuffer);
            _matmulKernel.SetMemoryArgument(2, o_t_buffer);
            _matmulKernel.SetValueArgument(3, 1);
            _matmulKernel.SetValueArgument(4, hiddenSize);
            _matmulKernel.SetValueArgument(5, hiddenSize);
            _queue.Execute(_matmulKernel, null, new long[] { 1, hiddenSize }, null, null);

            _elementwiseAddKernel.SetMemoryArgument(0, temp_buffer);
            _elementwiseAddKernel.SetMemoryArgument(1, o_t_buffer);
            _elementwiseAddKernel.SetMemoryArgument(2, o_t_buffer);
            _elementwiseAddKernel.SetValueArgument(3, hiddenSize);
            _queue.Execute(_elementwiseAddKernel, null, new long[] { hiddenSize }, null, null);

            _elementwiseAddKernel.SetMemoryArgument(0, o_t_buffer);
            _elementwiseAddKernel.SetMemoryArgument(1, boBuffer);
            _elementwiseAddKernel.SetMemoryArgument(2, o_t_buffer);
            _elementwiseAddKernel.SetValueArgument(3, hiddenSize);
            _queue.Execute(_elementwiseAddKernel, null, new long[] { hiddenSize }, null, null);

            _sigmoidKernel.SetMemoryArgument(0, o_t_buffer);
            _sigmoidKernel.SetMemoryArgument(1, o_t_buffer);
            _sigmoidKernel.SetValueArgument(2, hiddenSize);
            _queue.Execute(_sigmoidKernel, null, new long[] { hiddenSize }, null, null);

            GCHandle o_t_handle = GCHandle.Alloc(o_t_data, GCHandleType.Pinned);
            try
            {
                _queue.Read(o_t_buffer, true, 0, hiddenSize, o_t_handle.AddrOfPinnedObject(), null);
            }
            finally
            {
                o_t_handle.Free();
            }

            // Estado da célula: c_t = f_t * c_prev + i_t * c_tilde
            _elementwiseMultiplyKernel.SetMemoryArgument(0, f_t_buffer);
            _elementwiseMultiplyKernel.SetMemoryArgument(1, cPrevBuffer);
            _elementwiseMultiplyKernel.SetMemoryArgument(2, c_t_buffer);
            _elementwiseMultiplyKernel.SetValueArgument(3, hiddenSize);
            _queue.Execute(_elementwiseMultiplyKernel, null, new long[] { hiddenSize }, null, null);

            ComputeBuffer<float> i_c_tilde_buffer = new ComputeBuffer<float>(_context, ComputeMemoryFlags.WriteOnly, hiddenSize);
            _elementwiseMultiplyKernel.SetMemoryArgument(0, i_t_buffer);
            _elementwiseMultiplyKernel.SetMemoryArgument(1, c_tilde_buffer);
            _elementwiseMultiplyKernel.SetMemoryArgument(2, i_c_tilde_buffer);
            _elementwiseMultiplyKernel.SetValueArgument(3, hiddenSize);
            _queue.Execute(_elementwiseMultiplyKernel, null, new long[] { hiddenSize }, null, null);

            _elementwiseAddKernel.SetMemoryArgument(0, c_t_buffer);
            _elementwiseAddKernel.SetMemoryArgument(1, i_c_tilde_buffer);
            _elementwiseAddKernel.SetMemoryArgument(2, c_t_buffer);
            _elementwiseAddKernel.SetValueArgument(3, hiddenSize);
            _queue.Execute(_elementwiseAddKernel, null, new long[] { hiddenSize }, null, null);

            GCHandle c_t_handle = GCHandle.Alloc(c_t_data, GCHandleType.Pinned);
            try
            {
                _queue.Read(c_t_buffer, true, 0, hiddenSize, c_t_handle.AddrOfPinnedObject(), null);
            }
            finally
            {
                c_t_handle.Free();
            }

            // Estado oculto: h_t = o_t * tanh(c_t)
            _tanhKernel.SetMemoryArgument(0, c_t_buffer);
            _tanhKernel.SetMemoryArgument(1, temp_buffer);
            _tanhKernel.SetValueArgument(2, hiddenSize);
            _queue.Execute(_tanhKernel, null, new long[] { hiddenSize }, null, null);

            _elementwiseMultiplyKernel.SetMemoryArgument(0, o_t_buffer);
            _elementwiseMultiplyKernel.SetMemoryArgument(1, temp_buffer);
            _elementwiseMultiplyKernel.SetMemoryArgument(2, h_t_buffer);
            _elementwiseMultiplyKernel.SetValueArgument(3, hiddenSize);
            _queue.Execute(_elementwiseMultiplyKernel, null, new long[] { hiddenSize }, null, null);

            GCHandle h_t_handle = GCHandle.Alloc(h_t_data, GCHandleType.Pinned);
            try
            {
                _queue.Read(h_t_buffer, true, 0, hiddenSize, h_t_handle.AddrOfPinnedObject(), null);
            }
            finally
            {
                h_t_handle.Free();
            }

            Tensor i_t = new Tensor(i_t_data.Select(f => (double)f).ToArray(), new int[] { hiddenSize });
            Tensor f_t = new Tensor(f_t_data.Select(f => (double)f).ToArray(), new int[] { hiddenSize });
            Tensor c_tilde = new Tensor(c_tilde_data.Select(f => (double)f).ToArray(), new int[] { hiddenSize });
            Tensor o_t = new Tensor(o_t_data.Select(f => (double)f).ToArray(), new int[] { hiddenSize });
            Tensor c_t = new Tensor(c_t_data.Select(f => (double)f).ToArray(), new int[] { hiddenSize });
            Tensor h_t = new Tensor(h_t_data.Select(f => (double)f).ToArray(), new int[] { hiddenSize });

            xBuffer.Dispose();
            hPrevBuffer.Dispose();
            cPrevBuffer.Dispose();
            wiBuffer.Dispose();
            uiBuffer.Dispose();
            biBuffer.Dispose();
            wfBuffer.Dispose();
            ufBuffer.Dispose();
            bfBuffer.Dispose();
            wcBuffer.Dispose();
            ucBuffer.Dispose();
            bcBuffer.Dispose();
            woBuffer.Dispose();
            uoBuffer.Dispose();
            boBuffer.Dispose();
            i_t_buffer.Dispose();
            f_t_buffer.Dispose();
            c_tilde_buffer.Dispose();
            o_t_buffer.Dispose();
            c_t_buffer.Dispose();
            h_t_buffer.Dispose();
            temp_buffer.Dispose();
            i_c_tilde_buffer.Dispose();

            return (h_t, c_t, i_t, f_t, c_tilde, o_t);
        }

        public Tensor ForwardLogits(Tensor input)
        {
            if (input.shape.Length != 1 || input.shape[0] != inputSize)
            {
                throw new ArgumentException($"O tensor de entrada deve ser unidimensional com tamanho {inputSize}. Recebido: {input.shape[0]}.");
            }

            int vocabSize = inputSize / contextWindowSize;
            double[] inputData = input.GetData();
            Tensor[] inputSteps = new Tensor[contextWindowSize];
            for (int t = 0; t < contextWindowSize; t++)
            {
                double[] stepData = new double[vocabSize];
                Array.Copy(inputData, t * vocabSize, stepData, 0, vocabSize);
                inputSteps[t] = new Tensor(stepData, new int[] { vocabSize });
            }

            Tensor h_t = new Tensor(new double[hiddenSize], new int[] { hiddenSize });
            Tensor c_t = new Tensor(new double[hiddenSize], new int[] { hiddenSize });
            for (int t = 0; t < contextWindowSize; t++)
            {
                (h_t, c_t, _, _, _, _) = LSTMStep(inputSteps[t], h_t, c_t);
            }

            float[] h_t_data = h_t.GetData().Select(d => (float)d).ToArray();
            float[] w_out_data = W_out.GetData().Select(d => (float)d).ToArray();
            float[] b_out_data = b_out.GetData().Select(d => (float)d).ToArray();
            float[] logits = new float[outputSize];
            ComputeBuffer<float> h_t_buffer = new ComputeBuffer<float>(_context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, h_t_data);
            ComputeBuffer<float> w_out_buffer = new ComputeBuffer<float>(_context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, w_out_data);
            ComputeBuffer<float> logits_buffer = new ComputeBuffer<float>(_context, ComputeMemoryFlags.WriteOnly, outputSize);
            _matmulKernel.SetMemoryArgument(0, h_t_buffer);
            _matmulKernel.SetMemoryArgument(1, w_out_buffer);
            _matmulKernel.SetMemoryArgument(2, logits_buffer);
            _matmulKernel.SetValueArgument(3, 1);
            _matmulKernel.SetValueArgument(4, hiddenSize);
            _matmulKernel.SetValueArgument(5, outputSize);
            _queue.Execute(_matmulKernel, null, new long[] { 1, outputSize }, null, null);
            _queue.Finish();

            GCHandle logits_handle = GCHandle.Alloc(logits, GCHandleType.Pinned);
            try
            {
                _queue.Read(logits_buffer, true, 0, outputSize, logits_handle.AddrOfPinnedObject(), null);
            }
            finally
            {
                logits_handle.Free();
            }

            for (int i = 0; i < outputSize; i++)
            {
                logits[i] += b_out_data[i];
            }

            h_t_buffer.Dispose();
            w_out_buffer.Dispose();
            logits_buffer.Dispose();

            return new Tensor(logits.Select(f => (double)f).ToArray(), new int[] { outputSize });
        }

        public Tensor Forward(Tensor input)
        {
            Tensor logits = ForwardLogits(input);
            float[] logits_data = logits.GetData().Select(d => (float)d).ToArray();
            float[] probs = new float[outputSize];
            ComputeBuffer<float> logits_buffer = new ComputeBuffer<float>(_context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, logits_data);
            ComputeBuffer<float> probs_buffer = new ComputeBuffer<float>(_context, ComputeMemoryFlags.WriteOnly, outputSize);
            ComputeBuffer<float> work_buffer_max = new ComputeBuffer<float>(_context, ComputeMemoryFlags.WriteOnly, 1);
            ComputeBuffer<float> work_buffer_sum = new ComputeBuffer<float>(_context, ComputeMemoryFlags.WriteOnly, 1);
            //ComputeBuffer<int> debug_flag = new ComputeBuffer<int>(_context, ComputeMemoryFlags.WriteOnly, 1);
            int[] initial_debug_data = new int[] { 0 }; // Inicialize o array do lado do host para 0
            ComputeBuffer<int> debug_flag = new ComputeBuffer<int>(_context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.CopyHostPointer, initial_debug_data);

            _softmaxKernel.SetMemoryArgument(0, logits_buffer);
            _softmaxKernel.SetMemoryArgument(1, probs_buffer);
            _softmaxKernel.SetValueArgument(2, 1);
            _softmaxKernel.SetValueArgument(3, outputSize);
            _softmaxKernel.SetMemoryArgument(4, work_buffer_max);
            _softmaxKernel.SetMemoryArgument(5, work_buffer_sum);
            _softmaxKernel.SetMemoryArgument(6, debug_flag);
            _queue.Execute(_softmaxKernel, null, new long[] { 1 }, null, null);
            _queue.Finish(); // Adicione aqui

            GCHandle probs_handle = GCHandle.Alloc(probs, GCHandleType.Pinned);
            try
            {
                _queue.Read(probs_buffer, true, 0, outputSize, probs_handle.AddrOfPinnedObject(), null);
            }
            finally
            {
                probs_handle.Free();
            }

            int[] debug_data = new int[1];
            GCHandle debug_handle = GCHandle.Alloc(debug_data, GCHandleType.Pinned);
            try
            {
                _queue.Read(debug_flag, true, 0, 1, debug_handle.AddrOfPinnedObject(), null);
            }
            finally
            {
                debug_handle.Free();
            }

            if (debug_data[0] != 0)
            {
                Console.WriteLine($"Aviso: Problemas numéricos detectados no softmax (debug_flag: {debug_data[0]}).");
            }

            logits_buffer.Dispose();
            probs_buffer.Dispose();
            work_buffer_max.Dispose();
            work_buffer_sum.Dispose();
            debug_flag.Dispose();

            return new Tensor(probs.Select(f => (double)f).ToArray(), new int[] { outputSize });
        }

        public double TrainEpoch(Tensor[] inputs, Tensor[] targets, double learningRate)
        {
            double epochLoss = 0;

            for (int j = 0; j < inputs.Length; j++)
            {
                int vocabSize = inputSize / contextWindowSize;
                double[] inputData = inputs[j].GetData();
                Tensor[] inputSteps = new Tensor[contextWindowSize];
                for (int t = 0; t < contextWindowSize; t++)
                {
                    double[] stepData = new double[vocabSize];
                    Array.Copy(inputData, t * vocabSize, stepData, 0, vocabSize);
                    inputSteps[t] = new Tensor(stepData, new int[] { vocabSize });
                }

                Tensor h_t = new Tensor(new double[hiddenSize], new int[] { hiddenSize });
                Tensor c_t = new Tensor(new double[hiddenSize], new int[] { hiddenSize });
                Tensor[] i_ts = new Tensor[contextWindowSize];
                Tensor[] f_ts = new Tensor[contextWindowSize];
                Tensor[] c_tildes = new Tensor[contextWindowSize];
                Tensor[] c_ts = new Tensor[contextWindowSize];
                Tensor[] o_ts = new Tensor[contextWindowSize];
                Tensor[] h_ts = new Tensor[contextWindowSize];
                Tensor[] c_prevs = new Tensor[contextWindowSize];

                for (int t = 0; t < contextWindowSize; t++)
                {
                    Tensor h_prev = t == 0 ? new Tensor(new double[hiddenSize], new int[] { hiddenSize }) : h_ts[t - 1];
                    Tensor c_prev = t == 0 ? new Tensor(new double[hiddenSize], new int[] { hiddenSize }) : c_ts[t - 1];
                    (h_t, c_t, i_ts[t], f_ts[t], c_tildes[t], o_ts[t]) = LSTMStep(inputSteps[t], h_prev, c_prev);
                    h_ts[t] = new Tensor(h_t.GetData(), new int[] { hiddenSize });
                    c_ts[t] = new Tensor(c_t.GetData(), new int[] { hiddenSize });
                    c_prevs[t] = new Tensor(c_prev.GetData(), new int[] { hiddenSize });
                }

                Tensor output = Forward(inputs[j]);
                for (int o = 0; o < outputSize; o++)
                {
                    if (targets[j].Infer(new int[] { o }) == 1.0)
                    {
                        double outputValue = output.Infer(new int[] { o });
                        epochLoss += -Math.Log(outputValue + 1e-9);
                        break;
                    }
                }

                double[] grad_output = new double[outputSize];
                for (int o = 0; o < outputSize; o++)
                {
                    grad_output[o] = output.Infer(new int[] { o }) - targets[j].Infer(new int[] { o });
                }

                double[] grad_W_out = new double[hiddenSize * outputSize];
                double[] grad_b_out = new double[outputSize];
                for (int o = 0; o < outputSize; o++)
                {
                    for (int h = 0; h < hiddenSize; h++)
                    {
                        grad_W_out[h * outputSize + o] = grad_output[o] * h_ts[contextWindowSize - 1].Infer(new int[] { h });
                    }
                    grad_b_out[o] = grad_output[o];
                }

                double[] grad_h_next = new double[hiddenSize];
                for (int h = 0; h < hiddenSize; h++)
                {
                    double sum = 0;
                    for (int o = 0; o < outputSize; o++)
                    {
                        sum += grad_output[o] * W_out.Infer(new int[] { h, o });
                    }
                    grad_h_next[h] = sum;
                }

                double[] grad_c_next = new double[hiddenSize];
                double[][] grad_i_ts = new double[contextWindowSize][];
                double[][] grad_f_ts = new double[contextWindowSize][];
                double[][] grad_c_tildes = new double[contextWindowSize][];
                double[][] grad_o_ts = new double[contextWindowSize][];

                for (int t = contextWindowSize - 1; t >= 0; t--)
                {
                    grad_i_ts[t] = new double[hiddenSize];
                    grad_f_ts[t] = new double[hiddenSize];
                    grad_c_tildes[t] = new double[hiddenSize];
                    grad_o_ts[t] = new double[hiddenSize];

                    double[] grad_h_t = new double[hiddenSize];
                    for (int h = 0; h < hiddenSize; h++)
                    {
                        grad_h_t[h] = t == contextWindowSize - 1 ? grad_h_next[h] : 0;
                    }

                    double[] grad_c_t = new double[hiddenSize];
                    for (int h = 0; h < hiddenSize; h++)
                    {
                        grad_c_t[h] = grad_h_t[h] * o_ts[t].Infer(new int[] { h }) * TanhDerivative(Tanh(c_ts[t].Infer(new int[] { h }))) +
                                      (t < contextWindowSize - 1 ? grad_c_next[h] * f_ts[t + 1].Infer(new int[] { h }) : 0);
                    }

                    for (int h = 0; h < hiddenSize; h++)
                    {
                        grad_o_ts[t][h] = grad_h_t[h] * Tanh(c_ts[t].Infer(new int[] { h })) * SigmoidDerivative(o_ts[t].Infer(new int[] { h }));
                        grad_c_tildes[t][h] = grad_c_t[h] * i_ts[t].Infer(new int[] { h }) * TanhDerivative(c_tildes[t].Infer(new int[] { h }));
                        grad_i_ts[t][h] = grad_c_t[h] * c_tildes[t].Infer(new int[] { h }) * SigmoidDerivative(i_ts[t].Infer(new int[] { h }));
                        grad_f_ts[t][h] = grad_c_t[h] * c_prevs[t].Infer(new int[] { h }) * SigmoidDerivative(f_ts[t].Infer(new int[] { h }));
                    }

                    grad_c_next = grad_c_t;
                }

                double[] grad_W_i = new double[inputSize * hiddenSize];
                double[] grad_U_i = new double[hiddenSize * hiddenSize];
                double[] grad_b_i = new double[hiddenSize];
                double[] grad_W_f = new double[inputSize * hiddenSize];
                double[] grad_U_f = new double[hiddenSize * hiddenSize];
                double[] grad_b_f = new double[hiddenSize];
                double[] grad_W_c = new double[inputSize * hiddenSize];
                double[] grad_U_c = new double[hiddenSize * hiddenSize];
                double[] grad_b_c = new double[hiddenSize];
                double[] grad_W_o = new double[inputSize * hiddenSize];
                double[] grad_U_o = new double[hiddenSize * hiddenSize];
                double[] grad_b_o = new double[hiddenSize];

                for (int t = 0; t < contextWindowSize; t++)
                {
                    double[] input_t_data = inputSteps[t].GetData();
                    double[] h_prev_data = t == 0 ? new double[hiddenSize] : h_ts[t - 1].GetData();
                    for (int h = 0; h < hiddenSize; h++)
                    {
                        for (int i = 0; i < inputSteps[t].shape[0]; i++)
                        {
                            grad_W_i[i * hiddenSize + h] += grad_i_ts[t][h] * input_t_data[i];
                            grad_W_f[i * hiddenSize + h] += grad_f_ts[t][h] * input_t_data[i];
                            grad_W_c[i * hiddenSize + h] += grad_c_tildes[t][h] * input_t_data[i];
                            grad_W_o[i * hiddenSize + h] += grad_o_ts[t][h] * input_t_data[i];
                        }
                        for (int h_prev = 0; h_prev < hiddenSize; h_prev++)
                        {
                            grad_U_i[h_prev * hiddenSize + h] += grad_i_ts[t][h] * h_prev_data[h_prev];
                            grad_U_f[h_prev * hiddenSize + h] += grad_f_ts[t][h] * h_prev_data[h_prev];
                            grad_U_c[h_prev * hiddenSize + h] += grad_c_tildes[t][h] * h_prev_data[h_prev];
                            grad_U_o[h_prev * hiddenSize + h] += grad_o_ts[t][h] * h_prev_data[h_prev];
                        }
                        grad_b_i[h] += grad_i_ts[t][h];
                        grad_b_f[h] += grad_f_ts[t][h];
                        grad_b_c[h] += grad_c_tildes[t][h];
                        grad_b_o[h] += grad_o_ts[t][h];
                    }
                }

                UpdateWeights(W_i, grad_W_i, learningRate);
                UpdateWeights(U_i, grad_U_i, learningRate);
                UpdateWeights(b_i, grad_b_i, learningRate);
                UpdateWeights(W_f, grad_W_f, learningRate);
                UpdateWeights(U_f, grad_U_f, learningRate);
                UpdateWeights(b_f, grad_b_f, learningRate);
                UpdateWeights(W_c, grad_W_c, learningRate);
                UpdateWeights(U_c, grad_U_c, learningRate);
                UpdateWeights(b_c, grad_b_c, learningRate);
                UpdateWeights(W_o, grad_W_o, learningRate);
                UpdateWeights(U_o, grad_U_o, learningRate);
                UpdateWeights(b_o, grad_b_o, learningRate);
                UpdateWeights(W_out, grad_W_out, learningRate);
                UpdateWeights(b_out, grad_b_out, learningRate);
            }

            return epochLoss / inputs.Length;
        }

        private void UpdateWeights(Tensor tensor, double[] grad, double learningRate)
        {
            double[] data = tensor.GetData();
            double[] updatedData = new double[data.Length];
            for (int i = 0; i < data.Length; i++)
            {
                updatedData[i] = data[i] - learningRate * grad[i];
            }
            tensor.SetData(updatedData);
        }

        public void SaveModel(string filePath)
        {
            try
            {
                var modelData = new NeuralNetworkModelData
                {
                    InputSize = inputSize,
                    HiddenSize = hiddenSize,
                    OutputSize = outputSize,
                    WeightsInputGate = new TensorData { data = W_i.GetData(), shape = W_i.GetShape() },
                    RecurrentWeightsInputGate = new TensorData { data = U_i.GetData(), shape = U_i.GetShape() },
                    BiasInputGate = new TensorData { data = b_i.GetData(), shape = b_i.GetShape() },
                    WeightsForgetGate = new TensorData { data = W_f.GetData(), shape = W_f.GetShape() },
                    RecurrentWeightsForgetGate = new TensorData { data = U_f.GetData(), shape = U_f.GetShape() },
                    BiasForgetGate = new TensorData { data = b_f.GetData(), shape = b_f.GetShape() },
                    WeightsCellGate = new TensorData { data = W_c.GetData(), shape = W_c.GetShape() },
                    RecurrentWeightsCellGate = new TensorData { data = U_c.GetData(), shape = U_c.GetShape() },
                    BiasCellGate = new TensorData { data = b_c.GetData(), shape = b_c.GetShape() },
                    WeightsOutputGate = new TensorData { data = W_o.GetData(), shape = W_o.GetShape() },
                    RecurrentWeightsOutputGate = new TensorData { data = U_o.GetData(), shape = U_o.GetShape() },
                    BiasOutputGate = new TensorData { data = b_o.GetData(), shape = b_o.GetShape() },
                    WeightsOutput = new TensorData { data = W_out.GetData(), shape = W_out.GetShape() },
                    BiasOutput = new TensorData { data = b_out.GetData(), shape = b_out.GetShape() }
                };

                var options = new JsonSerializerOptions
                {
                    WriteIndented = true,
                    NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals
                };
                string jsonString = JsonSerializer.Serialize(modelData, options);
                File.WriteAllText(filePath, jsonString);
                Console.WriteLine($"Modelo salvo em JSON: {filePath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erro ao salvar o modelo: {ex.Message}");
            }
        }

        public static NeuralNetwork? LoadModel(string filePath)
        {
            try
            {
                if (!File.Exists(filePath))
                {
                    Console.WriteLine($"Arquivo do modelo não encontrado: {filePath}");
                    return null;
                }

                string jsonString = File.ReadAllText(filePath);
                var modelData = JsonSerializer.Deserialize<NeuralNetworkModelData>(jsonString);
                if (modelData == null || 
                    modelData.WeightsInputGate?.data == null || modelData.RecurrentWeightsInputGate?.data == null || modelData.BiasInputGate?.data == null ||
                    modelData.WeightsForgetGate?.data == null || modelData.RecurrentWeightsForgetGate?.data == null || modelData.BiasForgetGate?.data == null ||
                    modelData.WeightsCellGate?.data == null || modelData.RecurrentWeightsCellGate?.data == null || modelData.BiasCellGate?.data == null ||
                    modelData.WeightsOutputGate?.data == null || modelData.RecurrentWeightsOutputGate?.data == null || modelData.BiasOutputGate?.data == null ||
                    modelData.WeightsOutput?.data == null || modelData.BiasOutput?.data == null)
                {
                    throw new Exception("Dados do modelo JSON estão incompletos.");
                }

                Tensor? W_i = new Tensor(modelData.WeightsInputGate.data, modelData.WeightsInputGate.shape);
                Tensor? U_i = new Tensor(modelData.RecurrentWeightsInputGate.data, modelData.RecurrentWeightsInputGate.shape);
                Tensor? b_i = new Tensor(modelData.BiasInputGate.data, modelData.BiasInputGate.shape);
                Tensor? W_f = new Tensor(modelData.WeightsForgetGate.data, modelData.WeightsForgetGate.shape);
                Tensor? U_f = new Tensor(modelData.RecurrentWeightsForgetGate.data, modelData.RecurrentWeightsForgetGate.shape);
                Tensor? b_f = new Tensor(modelData.BiasForgetGate.data, modelData.BiasForgetGate.shape);
                Tensor? W_c = new Tensor(modelData.WeightsCellGate.data, modelData.WeightsCellGate.shape);
                Tensor? U_c = new Tensor(modelData.RecurrentWeightsCellGate.data, modelData.RecurrentWeightsCellGate.shape);
                Tensor? b_c = new Tensor(modelData.BiasCellGate.data, modelData.BiasCellGate.shape);
                Tensor? W_o = new Tensor(modelData.WeightsOutputGate.data, modelData.WeightsOutputGate.shape);
                Tensor? U_o = new Tensor(modelData.RecurrentWeightsOutputGate.data, modelData.RecurrentWeightsOutputGate.shape);
                Tensor? b_o = new Tensor(modelData.BiasOutputGate.data, modelData.BiasOutputGate.shape);
                Tensor? W_out = new Tensor(modelData.WeightsOutput.data, modelData.WeightsOutput.shape);
                Tensor? b_out = new Tensor(modelData.BiasOutput.data, modelData.BiasOutput.shape);

                return new NeuralNetwork(modelData.InputSize, modelData.HiddenSize, modelData.OutputSize, modelData.InputSize / modelData.OutputSize,
                    W_i, U_i, b_i, W_f, U_f, b_f, W_c, U_c, b_c, W_o, U_o, b_o, W_out, b_out);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erro ao carregar o modelo: {ex.Message}");
                return null;
            }
        }
    }
}