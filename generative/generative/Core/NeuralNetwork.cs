using OpenCL.Net;
using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Collections.Generic;
using System.Text;

namespace Core
{
    public class NeuralNetwork : IDisposable
    {
        private Tensor W_i, U_i, b_i;
        private Tensor W_f, U_f, b_f;
        private Tensor W_c, U_c, b_c;
        private Tensor W_o, U_o, b_o;
        private Tensor W_out, b_out;

        private readonly int inputSize, hiddenSize, outputSize, contextWindowSize;

        // OpenCL.NET objects
        private Context? _context;
        private Device? _device;
        private CommandQueue? _queue;
        private OpenCL.Net.Program? _program;
        private Kernel? _matmulKernel;
        private Kernel? _sigmoidKernel;
        private Kernel? _tanhKernel;
        private Kernel? _elementwiseAddKernel;
        private Kernel? _elementwiseAddBroadcastKernel;
        private Kernel? _elementwiseMultiplyKernel;

        // Gradient clipping threshold
        private const double GRADIENT_CLIP_THRESHOLD = 5.0;

        // Learning rate decay parameters
        private double _currentLearningRate;
        private const double LEARNING_RATE_DECAY = 0.95;
        private const int DECAY_EVERY_N_EPOCHS = 10;

        public Tensor W_i_Tensor => W_i;
        public Tensor U_i_Tensor => U_i;
        public Tensor b_i_Tensor => b_i;
        public Tensor W_f_Tensor => W_f;
        public Tensor U_f_Tensor => U_f;
        public Tensor b_f_Tensor => b_f;
        public Tensor W_c_Tensor => W_c;
        public Tensor U_c_Tensor => U_c;
        public Tensor b_c_Tensor => b_c;
        public Tensor W_o_Tensor => W_o;
        public Tensor U_o_Tensor => U_o;
        public Tensor b_o_Tensor => b_o;
        public Tensor W_out_Tensor => W_out;
        public Tensor b_out_Tensor => b_out;

        public int InputSize => inputSize;
        public int HiddenSize => hiddenSize;
        public int OutputSize => outputSize;
        public bool IsGpuEnabled => _context.HasValue;

        public NeuralNetwork(int inputSize, int hiddenSize, int outputSize, int contextWindowSize,
            double initialLearningRate = 0.001)
        {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;
            this.contextWindowSize = contextWindowSize;
            this._currentLearningRate = initialLearningRate;

            InitializeWeights();
            InitializeOpenCL();
        }

        private NeuralNetwork(int inputSize, int hiddenSize, int outputSize, int contextWindowSize,
            Tensor W_i, Tensor U_i, Tensor b_i, Tensor W_f, Tensor U_f, Tensor b_f,
            Tensor W_c, Tensor U_c, Tensor b_c, Tensor W_o, Tensor U_o, Tensor b_o,
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

        private void InitializeWeights()
        {
            int vocabSize = outputSize;
            Random rand = new Random();

            // Xavier/Glorot initialization for better convergence
            double sqrtFanInHidden = Math.Sqrt(6.0 / (vocabSize + hiddenSize));
            double sqrtFanInRecurrent = Math.Sqrt(6.0 / (2 * hiddenSize));
            double sqrtFanInOutput = Math.Sqrt(6.0 / (hiddenSize + outputSize));

            W_i = new Tensor(InitializeWeights(vocabSize, hiddenSize, sqrtFanInHidden, rand),
                new int[] { vocabSize, hiddenSize });
            U_i = new Tensor(InitializeWeights(hiddenSize, hiddenSize, sqrtFanInRecurrent, rand),
                new int[] { hiddenSize, hiddenSize });
            b_i = new Tensor(new double[hiddenSize], new int[] { hiddenSize });

            W_f = new Tensor(InitializeWeights(vocabSize, hiddenSize, sqrtFanInHidden, rand),
                new int[] { vocabSize, hiddenSize });
            U_f = new Tensor(InitializeWeights(hiddenSize, hiddenSize, sqrtFanInRecurrent, rand),
                new int[] { hiddenSize, hiddenSize });
            // Initialize forget gate bias to 1.0 for better gradient flow
            double[] forgetBias = new double[hiddenSize];
            for (int i = 0; i < hiddenSize; i++) forgetBias[i] = 1.0;
            b_f = new Tensor(forgetBias, new int[] { hiddenSize });

            W_c = new Tensor(InitializeWeights(vocabSize, hiddenSize, sqrtFanInHidden, rand),
                new int[] { vocabSize, hiddenSize });
            U_c = new Tensor(InitializeWeights(hiddenSize, hiddenSize, sqrtFanInRecurrent, rand),
                new int[] { hiddenSize, hiddenSize });
            b_c = new Tensor(new double[hiddenSize], new int[] { hiddenSize });

            W_o = new Tensor(InitializeWeights(vocabSize, hiddenSize, sqrtFanInHidden, rand),
                new int[] { vocabSize, hiddenSize });
            U_o = new Tensor(InitializeWeights(hiddenSize, hiddenSize, sqrtFanInRecurrent, rand),
                new int[] { hiddenSize, hiddenSize });
            b_o = new Tensor(new double[hiddenSize], new int[] { hiddenSize });

            W_out = new Tensor(InitializeWeights(hiddenSize, outputSize, sqrtFanInOutput, rand),
                new int[] { hiddenSize, outputSize });
            b_out = new Tensor(new double[outputSize], new int[] { outputSize });
        }

        private void InitializeOpenCL()
        {
            try
            {
                ErrorCode error;

                Cl.GetPlatformIDs(0, null, out uint numPlatforms);
                if (numPlatforms == 0)
                {
                    Console.WriteLine("Nenhuma plataforma OpenCL encontrada.");
                    return;
                }

                Platform[] platforms = new Platform[numPlatforms];
                Cl.GetPlatformIDs(numPlatforms, platforms, out numPlatforms);

                Platform selectedPlatform = platforms[0];

                Cl.GetDeviceIDs(selectedPlatform, DeviceType.Gpu, 0, null, out uint numDevices);

                Device[] devices;
                if (numDevices == 0)
                {
                    Cl.GetDeviceIDs(selectedPlatform, DeviceType.Cpu, 0, null, out numDevices);
                    if (numDevices == 0)
                    {
                        Console.WriteLine("Nenhum dispositivo OpenCL encontrado.");
                        return;
                    }

                    devices = new Device[numDevices];
                    Cl.GetDeviceIDs(selectedPlatform, DeviceType.Cpu, numDevices, devices, out numDevices);
                }
                else
                {
                    devices = new Device[numDevices];
                    Cl.GetDeviceIDs(selectedPlatform, DeviceType.Gpu, numDevices, devices, out numDevices);
                }

                _device = devices[0];

                InfoBuffer deviceNameBuffer = Cl.GetDeviceInfo(_device.Value, DeviceInfo.Name, out error);
                string deviceName = deviceNameBuffer.ToString();

                InfoBuffer platformNameBuffer = Cl.GetPlatformInfo(selectedPlatform, PlatformInfo.Name, out error);
                string platformName = platformNameBuffer.ToString();

                Console.WriteLine($"Usando dispositivo OpenCL: {deviceName} da plataforma {platformName}");

                _context = Cl.CreateContext(null, 1, new Device[] { _device.Value }, null, IntPtr.Zero, out error);
                CheckError(error, "CreateContext");

                _queue = Cl.CreateCommandQueue(_context.Value, _device.Value, CommandQueueProperties.None, out error);
                CheckError(error, "CreateCommandQueue");

                string kernelPath = Path.Combine(AppContext.BaseDirectory, "Kernels", "MatrixOperations.cl");
                if (!File.Exists(kernelPath))
                {
                    kernelPath =
                        "/home/mplopes/Documentos/GitHub/gen.AI/generative/generative/Kernels/MatrixOperations.cl";
                    if (!File.Exists(kernelPath))
                        throw new FileNotFoundException($"Arquivo de kernel OpenCL não encontrado: {kernelPath}");
                }

                string kernelSource = File.ReadAllText(kernelPath);
                _program = Cl.CreateProgramWithSource(_context.Value, 1, new[] { kernelSource }, null, out error);
                CheckError(error, "CreateProgramWithSource");

                error = Cl.BuildProgram(_program.Value, 1, new Device[] { _device.Value }, string.Empty, null,
                    IntPtr.Zero);
                if (error != ErrorCode.Success)
                {
                    InfoBuffer buildLog =
                        Cl.GetProgramBuildInfo(_program.Value, _device.Value, ProgramBuildInfo.Log, out error);
                    Console.WriteLine($"Build Log: {buildLog}");
                    throw new Exception($"Erro ao compilar programa OpenCL: {error}");
                }

                _matmulKernel = Cl.CreateKernel(_program.Value, "matmul_forward", out error);
                CheckError(error, "CreateKernel matmul_forward");

                _sigmoidKernel = Cl.CreateKernel(_program.Value, "sigmoid_forward", out error);
                CheckError(error, "CreateKernel sigmoid_forward");

                _tanhKernel = Cl.CreateKernel(_program.Value, "tanh_forward", out error);
                CheckError(error, "CreateKernel tanh_forward");

                _elementwiseAddKernel = Cl.CreateKernel(_program.Value, "elementwise_add_forward", out error);
                CheckError(error, "CreateKernel elementwise_add_forward");

                _elementwiseAddBroadcastKernel =
                    Cl.CreateKernel(_program.Value, "elementwise_add_broadcast_forward", out error);
                CheckError(error, "CreateKernel elementwise_add_broadcast_forward");

                _elementwiseMultiplyKernel = Cl.CreateKernel(_program.Value, "elementwise_multiply", out error);
                CheckError(error, "CreateKernel elementwise_multiply");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erro ao inicializar OpenCL: {ex.Message}. Revertendo para CPU.");
                CleanupOpenCL();
            }
        }

        private void CheckError(ErrorCode error, string operation = "")
        {
            if (error != ErrorCode.Success)
            {
                throw new Exception($"OpenCL Error on operation '{operation}': {error}");
            }
        }


        private Tensor ExecuteMatMulGpu(Tensor A, Tensor B)
        {
            if (!_context.HasValue || !_queue.HasValue || !_matmulKernel.HasValue)
                throw new InvalidOperationException("OpenCL não está inicializado corretamente.");

            Tensor A_expanded = (A.shape.Length == 1) ? new Tensor(A.GetData(), new int[] { 1, A.shape[0] }) : A;
            Tensor B_expanded = (B.shape.Length == 1) ? new Tensor(B.GetData(), new int[] { B.shape[0], 1 }) : B;

            int M = A_expanded.shape[0];
            int K = A_expanded.shape[1];
            // CORREÇÃO CRÍTICA FINAL: N deve ser a segunda dimensão de B, não de B_expanded
            int N = B.shape.Length > 1 ? B.shape[1] : 1;
            if (B.shape.Length == 1 && A.shape.Length > 1) N = 1; // Matrix * Vector
            if (B.shape.Length > 1) N = B.shape[1]; // Matrix * Matrix or Vector * Matrix

            double[] resultData = new double[M * N];
            int[] resultShape;

            if (A.shape.Length == 1 && B.shape.Length == 1) resultShape = new int[] { 1 };
            else if (A.shape.Length == 1) resultShape = new int[] { N };
            else if (B.shape.Length == 1) resultShape = new int[] { M };
            else resultShape = new int[] { M, N };

            ErrorCode error;

            IMem bufferA = Cl.CreateBuffer(_context.Value, MemFlags.ReadOnly | MemFlags.CopyHostPtr,
                (IntPtr)(A_expanded.GetData().Length * sizeof(double)), A_expanded.GetData(), out error);
            CheckError(error, "CreateBuffer A");
            IMem bufferB = Cl.CreateBuffer(_context.Value, MemFlags.ReadOnly | MemFlags.CopyHostPtr,
                (IntPtr)(B_expanded.GetData().Length * sizeof(double)), B_expanded.GetData(), out error);
            CheckError(error, "CreateBuffer B");
            IMem bufferC = Cl.CreateBuffer(_context.Value, MemFlags.WriteOnly,
                (IntPtr)(resultData.Length * sizeof(double)), IntPtr.Zero, out error);
            CheckError(error, "CreateBuffer C");

            try
            {
                error = Cl.SetKernelArg(_matmulKernel.Value, 0, bufferA);
                CheckError(error);
                error = Cl.SetKernelArg(_matmulKernel.Value, 1, bufferB);
                CheckError(error);
                error = Cl.SetKernelArg(_matmulKernel.Value, 2, bufferC);
                CheckError(error);
                error = Cl.SetKernelArg(_matmulKernel.Value, 3, M);
                CheckError(error);
                error = Cl.SetKernelArg(_matmulKernel.Value, 4, K);
                CheckError(error);
                error = Cl.SetKernelArg(_matmulKernel.Value, 5, N);
                CheckError(error);

                IntPtr[] globalWorkSize = new IntPtr[] { (IntPtr)N, (IntPtr)M };
                error = Cl.EnqueueNDRangeKernel(_queue.Value, _matmulKernel.Value, 2, null, globalWorkSize, null, 0,
                    null, out Event ev);
                CheckError(error, "EnqueueNDRangeKernel MatMul");
                Cl.WaitForEvents(1, new Event[] { ev });
                ev.Dispose();

                error = Cl.EnqueueReadBuffer(_queue.Value, bufferC, Bool.True, IntPtr.Zero,
                    (IntPtr)(resultData.Length * sizeof(double)), resultData, 0, null, out ev);
                CheckError(error, "EnqueueReadBuffer MatMul");
                Cl.WaitForEvents(1, new Event[] { ev });
                ev.Dispose();
            }
            finally
            {
                Cl.ReleaseMemObject(bufferA);
                Cl.ReleaseMemObject(bufferB);
                Cl.ReleaseMemObject(bufferC);
            }

            return new Tensor(resultData, resultShape);
        }

        private Tensor ExecuteElementwiseGpu(Kernel kernel, Tensor A, Tensor B)
        {
            if (!_context.HasValue || !_queue.HasValue)
                throw new InvalidOperationException("OpenCL não está inicializado corretamente.");

            if (!A.shape.SequenceEqual(B.shape))
                throw new ArgumentException("Shapes devem ser idênticos para esta operação de GPU.");

            double[] resultData = new double[A.GetTotalSize()];
            ErrorCode error;

            IMem bufferA = Cl.CreateBuffer(_context.Value, MemFlags.ReadOnly | MemFlags.CopyHostPtr,
                (IntPtr)(A.GetData().Length * sizeof(double)), A.GetData(), out error);
            CheckError(error, "CreateBuffer A");
            IMem bufferB = Cl.CreateBuffer(_context.Value, MemFlags.ReadOnly | MemFlags.CopyHostPtr,
                (IntPtr)(B.GetData().Length * sizeof(double)), B.GetData(), out error);
            CheckError(error, "CreateBuffer B");
            IMem bufferC = Cl.CreateBuffer(_context.Value, MemFlags.WriteOnly,
                (IntPtr)(resultData.Length * sizeof(double)), IntPtr.Zero, out error);
            CheckError(error, "CreateBuffer C");

            try
            {
                error = Cl.SetKernelArg(kernel, 0, bufferA);
                CheckError(error);
                error = Cl.SetKernelArg(kernel, 1, bufferB);
                CheckError(error);
                error = Cl.SetKernelArg(kernel, 2, bufferC);
                CheckError(error);
                error = Cl.SetKernelArg(kernel, 3, A.GetTotalSize());
                CheckError(error);

                IntPtr[] globalWorkSize = new IntPtr[] { (IntPtr)A.GetTotalSize() };
                error = Cl.EnqueueNDRangeKernel(_queue.Value, kernel, 1, null, globalWorkSize, null, 0, null,
                    out Event ev);
                CheckError(error, "EnqueueNDRangeKernel Elementwise");
                Cl.WaitForEvents(1, new Event[] { ev });
                ev.Dispose();

                error = Cl.EnqueueReadBuffer(_queue.Value, bufferC, Bool.True, IntPtr.Zero,
                    (IntPtr)(resultData.Length * sizeof(double)), resultData, 0, null, out ev);
                CheckError(error, "EnqueueReadBuffer Elementwise");
                Cl.WaitForEvents(1, new Event[] { ev });
                ev.Dispose();
            }
            finally
            {
                Cl.ReleaseMemObject(bufferA);
                Cl.ReleaseMemObject(bufferB);
                Cl.ReleaseMemObject(bufferC);
            }

            return new Tensor(resultData, A.GetShape());
        }

        private Tensor ExecuteActivationGpu(Kernel kernel, Tensor A)
        {
            if (!_context.HasValue || !_queue.HasValue)
                throw new InvalidOperationException("OpenCL não está inicializado corretamente.");

            double[] resultData = new double[A.GetTotalSize()];
            ErrorCode error;

            IMem bufferIn = Cl.CreateBuffer(_context.Value, MemFlags.ReadOnly | MemFlags.CopyHostPtr,
                (IntPtr)(A.GetData().Length * sizeof(double)), A.GetData(), out error);
            CheckError(error);
            IMem bufferOut = Cl.CreateBuffer(_context.Value, MemFlags.WriteOnly,
                (IntPtr)(resultData.Length * sizeof(double)), IntPtr.Zero, out error);
            CheckError(error);

            try
            {
                error = Cl.SetKernelArg(kernel, 0, bufferIn);
                CheckError(error);
                error = Cl.SetKernelArg(kernel, 1, bufferOut);
                CheckError(error);
                error = Cl.SetKernelArg(kernel, 2, A.GetTotalSize());
                CheckError(error);

                IntPtr[] globalWorkSize = new IntPtr[] { (IntPtr)A.GetTotalSize() };
                error = Cl.EnqueueNDRangeKernel(_queue.Value, kernel, 1, null, globalWorkSize, null, 0, null,
                    out Event ev);
                CheckError(error, "EnqueueNDRangeKernel Activation");
                Cl.WaitForEvents(1, new Event[] { ev });
                ev.Dispose();

                error = Cl.EnqueueReadBuffer(_queue.Value, bufferOut, Bool.True, IntPtr.Zero,
                    (IntPtr)(resultData.Length * sizeof(double)), resultData, 0, null, out ev);
                CheckError(error, "EnqueueReadBuffer Activation");
                Cl.WaitForEvents(1, new Event[] { ev });
                ev.Dispose();
            }
            finally
            {
                Cl.ReleaseMemObject(bufferIn);
                Cl.ReleaseMemObject(bufferOut);
            }

            return new Tensor(resultData, A.GetShape());
        }

        private Tensor ExecuteElementwiseAddBroadcastGpu(Tensor A, Tensor B_vec)
        {
            if (!_context.HasValue || !_queue.HasValue || !_elementwiseAddBroadcastKernel.HasValue)
                throw new InvalidOperationException("OpenCL não está inicializado corretamente.");

            Console.WriteLine(
                $"Broadcasting - A: [{string.Join(", ", A.GetShape())}], B: [{string.Join(", ", B_vec.GetShape())}]");

            // Handle 1D tensor A by treating it as a row vector [1, N]
            Tensor A_for_broadcast;
            bool was_A_1D = A.shape.Length == 1;

            if (was_A_1D)
            {
                A_for_broadcast = new Tensor(A.GetData(), new int[] { 1, A.shape[0] });
            }
            else if (A.shape.Length == 2)
            {
                A_for_broadcast = A;
            }
            else
            {
                throw new ArgumentException(
                    $"Tensor A deve ser 1D ou 2D. Recebido shape: [{string.Join(", ", A.shape)}]");
            }

            // Ensure B is 1D
            if (B_vec.shape.Length != 1)
            {
                throw new ArgumentException(
                    $"Tensor B deve ser 1D. Recebido shape: [{string.Join(", ", B_vec.shape)}]");
            }

            int M = A_for_broadcast.shape[0];
            int N = A_for_broadcast.shape[1];

            // Check if broadcasting is valid
            if (N != B_vec.shape[0])
            {
                throw new ArgumentException(
                    $"Shapes incompatíveis para broadcasting. A: [{M},{N}], B: [{B_vec.shape[0]}]");
            }

            double[] resultData = new double[M * N];
            ErrorCode error;

            IMem bufferA = Cl.CreateBuffer(_context.Value, MemFlags.ReadOnly | MemFlags.CopyHostPtr,
                (IntPtr)(A_for_broadcast.GetData().Length * sizeof(double)), A_for_broadcast.GetData(), out error);
            CheckError(error);

            IMem bufferB = Cl.CreateBuffer(_context.Value, MemFlags.ReadOnly | MemFlags.CopyHostPtr,
                (IntPtr)(B_vec.GetData().Length * sizeof(double)), B_vec.GetData(), out error);
            CheckError(error);

            IMem bufferC = Cl.CreateBuffer(_context.Value, MemFlags.WriteOnly,
                (IntPtr)(resultData.Length * sizeof(double)), IntPtr.Zero, out error);
            CheckError(error);

            try
            {
                error = Cl.SetKernelArg(_elementwiseAddBroadcastKernel.Value, 0, bufferA);
                CheckError(error);
                error = Cl.SetKernelArg(_elementwiseAddBroadcastKernel.Value, 1, bufferB);
                CheckError(error);
                error = Cl.SetKernelArg(_elementwiseAddBroadcastKernel.Value, 2, bufferC);
                CheckError(error);
                error = Cl.SetKernelArg(_elementwiseAddBroadcastKernel.Value, 3, M);
                CheckError(error);
                error = Cl.SetKernelArg(_elementwiseAddBroadcastKernel.Value, 4, N);
                CheckError(error);

                IntPtr[] globalWorkSize = new IntPtr[] { (IntPtr)N, (IntPtr)M };
                error = Cl.EnqueueNDRangeKernel(_queue.Value, _elementwiseAddBroadcastKernel.Value, 2, null,
                    globalWorkSize, null, 0, null, out Event ev);
                CheckError(error, "EnqueueNDRangeKernel BroadcastAdd");
                Cl.WaitForEvents(1, new Event[] { ev });
                ev.Dispose();

                error = Cl.EnqueueReadBuffer(_queue.Value, bufferC, Bool.True, IntPtr.Zero,
                    (IntPtr)(resultData.Length * sizeof(double)), resultData, 0, null, out ev);
                CheckError(error, "EnqueueReadBuffer BroadcastAdd");
                Cl.WaitForEvents(1, new Event[] { ev });
                ev.Dispose();
            }
            finally
            {
                Cl.ReleaseMemObject(bufferA);
                Cl.ReleaseMemObject(bufferB);
                Cl.ReleaseMemObject(bufferC);
            }

            // Return tensor with the same shape as the original A tensor
            int[] resultShape = was_A_1D ? new int[] { N } : new int[] { M, N };
            Console.WriteLine($"Broadcast result shape: [{string.Join(", ", resultShape)}]");

            return new Tensor(resultData, resultShape);
        }

        private void LogTensorShape(string name, Tensor tensor)
        {
            Console.WriteLine(
                $"Tensor {name}: shape=[{string.Join(", ", tensor.GetShape())}], size={tensor.GetTotalSize()}");
        }

        private (Tensor h_t, Tensor c_t, Tensor i_t, Tensor f_t, Tensor c_tilde, Tensor o_t) LSTMStep(Tensor x_t,
            Tensor h_prev, Tensor c_prev)
        {
            if (!IsGpuEnabled)
            {
                // CPU fallback (unchanged)
                Tensor matmul_Wi_xt = x_t.MatMul(W_i);
                Tensor matmul_Ui_hprev = h_prev.MatMul(U_i);
                Tensor i_t_cpu = matmul_Wi_xt.Add(matmul_Ui_hprev).Add(b_i).Apply(Sigmoid);
                Tensor matmul_Wf_xt = x_t.MatMul(W_f);
                Tensor matmul_Uf_hprev = h_prev.MatMul(U_f);
                Tensor f_t_cpu = matmul_Wf_xt.Add(matmul_Uf_hprev).Add(b_f).Apply(Sigmoid);
                Tensor matmul_Wc_xt = x_t.MatMul(W_c);
                Tensor matmul_Uc_hprev = h_prev.MatMul(U_c);
                Tensor c_tilde_cpu = matmul_Wc_xt.Add(matmul_Uc_hprev).Add(b_c).Apply(Tanh);
                Tensor matmul_Wo_xt = x_t.MatMul(W_o);
                Tensor matmul_Uo_hprev = h_prev.MatMul(U_o);
                Tensor o_t_cpu = matmul_Wo_xt.Add(matmul_Uo_hprev).Add(b_o).Apply(Sigmoid);
                Tensor c_t_cpu = f_t_cpu.ElementWiseMultiply(c_prev).Add(i_t_cpu.ElementWiseMultiply(c_tilde_cpu));
                Tensor h_t_cpu = o_t_cpu.ElementWiseMultiply(c_t_cpu.Apply(Tanh));
                return (h_t_cpu, c_t_cpu, i_t_cpu, f_t_cpu, c_tilde_cpu, o_t_cpu);
            }

            try
            {
                // GPU execution
                Tensor matmul_Wi_xt_gpu = ExecuteMatMulGpu(x_t, W_i);
                Tensor matmul_Ui_hprev_gpu = ExecuteMatMulGpu(h_prev, U_i);
                Tensor add1_i =
                    ExecuteElementwiseGpu(_elementwiseAddKernel!.Value, matmul_Wi_xt_gpu, matmul_Ui_hprev_gpu);
                Tensor add2_i =
                    ExecuteElementwiseGpu(_elementwiseAddKernel!.Value, add1_i, b_i); // CORREÇÃO: Usar soma de vetores
                Tensor i_t = ExecuteActivationGpu(_sigmoidKernel!.Value, add2_i);

                Tensor matmul_Wf_xt_gpu = ExecuteMatMulGpu(x_t, W_f);
                Tensor matmul_Uf_hprev_gpu = ExecuteMatMulGpu(h_prev, U_f);
                Tensor add1_f =
                    ExecuteElementwiseGpu(_elementwiseAddKernel!.Value, matmul_Wf_xt_gpu, matmul_Uf_hprev_gpu);
                Tensor add2_f =
                    ExecuteElementwiseGpu(_elementwiseAddKernel!.Value, add1_f, b_f); // CORREÇÃO: Usar soma de vetores
                Tensor f_t = ExecuteActivationGpu(_sigmoidKernel!.Value, add2_f);

                Tensor matmul_Wc_xt_gpu = ExecuteMatMulGpu(x_t, W_c);
                Tensor matmul_Uc_hprev_gpu = ExecuteMatMulGpu(h_prev, U_c);
                Tensor add1_c =
                    ExecuteElementwiseGpu(_elementwiseAddKernel!.Value, matmul_Wc_xt_gpu, matmul_Uc_hprev_gpu);
                Tensor add2_c =
                    ExecuteElementwiseGpu(_elementwiseAddKernel!.Value, add1_c, b_c); // CORREÇÃO: Usar soma de vetores
                Tensor c_tilde = ExecuteActivationGpu(_tanhKernel!.Value, add2_c);

                Tensor matmul_Wo_xt_gpu = ExecuteMatMulGpu(x_t, W_o);
                Tensor matmul_Uo_hprev_gpu = ExecuteMatMulGpu(h_prev, U_o);
                Tensor add1_o =
                    ExecuteElementwiseGpu(_elementwiseAddKernel!.Value, matmul_Wo_xt_gpu, matmul_Uo_hprev_gpu);
                Tensor add2_o =
                    ExecuteElementwiseGpu(_elementwiseAddKernel!.Value, add1_o, b_o); // CORREÇÃO: Usar soma de vetores
                Tensor o_t = ExecuteActivationGpu(_sigmoidKernel!.Value, add2_o);

                Tensor f_mul_c_prev = ExecuteElementwiseGpu(_elementwiseMultiplyKernel!.Value, f_t, c_prev);
                Tensor i_mul_c_tilde = ExecuteElementwiseGpu(_elementwiseMultiplyKernel!.Value, i_t, c_tilde);
                Tensor c_t = ExecuteElementwiseGpu(_elementwiseAddKernel!.Value, f_mul_c_prev, i_mul_c_tilde);

                Tensor tanh_c_t = ExecuteActivationGpu(_tanhKernel!.Value, c_t);
                Tensor h_t = ExecuteElementwiseGpu(_elementwiseMultiplyKernel!.Value, o_t, tanh_c_t);

                return (h_t, c_t, i_t, f_t, c_tilde, o_t);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"GPU operation failed in LSTMStep: {ex.Message}");
                Console.WriteLine("Falling back to CPU execution...");
                // Fallback to CPU implementation
                Tensor matmul_Wi_xt = x_t.MatMul(W_i);
                Tensor matmul_Ui_hprev = h_prev.MatMul(U_i);
                Tensor i_t_cpu = matmul_Wi_xt.Add(matmul_Ui_hprev).Add(b_i).Apply(Sigmoid);
                Tensor matmul_Wf_xt = x_t.MatMul(W_f);
                Tensor matmul_Uf_hprev = h_prev.MatMul(U_f);
                Tensor f_t_cpu = matmul_Wf_xt.Add(matmul_Uf_hprev).Add(b_f).Apply(Sigmoid);
                Tensor matmul_Wc_xt = x_t.MatMul(W_c);
                Tensor matmul_Uc_hprev = h_prev.MatMul(U_c);
                Tensor c_tilde_cpu = matmul_Wc_xt.Add(matmul_Uc_hprev).Add(b_c).Apply(Tanh);
                Tensor matmul_Wo_xt = x_t.MatMul(W_o);
                Tensor matmul_Uo_hprev = h_prev.MatMul(U_o);
                Tensor o_t_cpu = matmul_Wo_xt.Add(matmul_Uo_hprev).Add(b_o).Apply(Sigmoid);
                Tensor c_t_cpu = f_t_cpu.ElementWiseMultiply(c_prev).Add(i_t_cpu.ElementWiseMultiply(c_tilde_cpu));
                Tensor h_t_cpu = o_t_cpu.ElementWiseMultiply(c_t_cpu.Apply(Tanh));
                return (h_t_cpu, c_t_cpu, i_t_cpu, f_t_cpu, c_tilde_cpu, o_t_cpu);
            }
        }

        public Tensor ForwardLogits(Tensor input)
        {
            if (input.shape.Length != 1 || input.shape[0] != inputSize)
            {
                throw new ArgumentException(
                    $"O tensor de entrada deve ser unidimensional com tamanho {inputSize}. Recebido: {input.shape[0]}.");
            }

            int vocabSize = outputSize;
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

            if (IsGpuEnabled)
            {
                Tensor matmul_Wout_ht = ExecuteMatMulGpu(h_t, W_out);
                // CORREÇÃO: Usar a soma de vetores, não broadcasting
                return ExecuteElementwiseGpu(_elementwiseAddKernel!.Value, matmul_Wout_ht, b_out);
            }
            else
            {
                Tensor matmul_Wout_ht = h_t.MatMul(W_out);
                return matmul_Wout_ht.Add(b_out);
            }
        }

        public Tensor Forward(Tensor input)
        {
            Tensor logits = ForwardLogits(input);
            return ApplySoftmax(logits);
        }

        private Tensor ApplySoftmax(Tensor logits)
        {
            double[] outputData = logits.GetData();
            double maxLogit = outputData.Max();
            double sumExp = 0;

            for (int o = 0; o < outputSize; o++)
            {
                outputData[o] = Math.Exp(outputData[o] - maxLogit);
                sumExp += outputData[o];
            }

            if (sumExp == 0) sumExp = 1e-12;

            for (int o = 0; o < outputSize; o++)
            {
                outputData[o] /= sumExp;
            }

            return new Tensor(outputData, new int[] { outputSize });
        }

        private double[] ClipGradients(double[] gradients)
        {
            double gradNorm = Math.Sqrt(gradients.Sum(g => g * g));

            if (gradNorm > GRADIENT_CLIP_THRESHOLD)
            {
                double clipRatio = GRADIENT_CLIP_THRESHOLD / gradNorm;
                for (int i = 0; i < gradients.Length; i++)
                {
                    gradients[i] *= clipRatio;
                }
            }

            return gradients;
        }

        public double TrainEpoch(List<(Tensor input, Tensor target)> dataset, double learningRate, int epoch = 0)
        {
            if (epoch > 0 && epoch % DECAY_EVERY_N_EPOCHS == 0)
            {
                _currentLearningRate *= LEARNING_RATE_DECAY;
                Console.WriteLine($"Learning rate decayed to: {_currentLearningRate:F6}");
            }

            double epochLoss = 0;
            int vocabSize = outputSize;

            // --- MUDANÇA: Inicializar acumuladores de gradientes AQUI (fora do loop) ---
            double[] grad_W_i_acc = new double[vocabSize * hiddenSize];
            double[] grad_U_i_acc = new double[hiddenSize * hiddenSize];
            double[] grad_b_i_acc = new double[hiddenSize];
            double[] grad_W_f_acc = new double[vocabSize * hiddenSize];
            double[] grad_U_f_acc = new double[hiddenSize * hiddenSize];
            double[] grad_b_f_acc = new double[hiddenSize];
            double[] grad_W_c_acc = new double[vocabSize * hiddenSize];
            double[] grad_U_c_acc = new double[hiddenSize * hiddenSize];
            double[] grad_b_c_acc = new double[hiddenSize];
            double[] grad_W_o_acc = new double[vocabSize * hiddenSize];
            double[] grad_U_o_acc = new double[hiddenSize * hiddenSize];
            double[] grad_b_o_acc = new double[hiddenSize];
            double[] grad_W_out_data_acc = new double[hiddenSize * outputSize];
            double[] grad_b_out_data_acc = new double[outputSize];

            foreach (var (input, target) in dataset)
            {
                // --- MUDANÇA: Não limpar os acumuladores aqui ---

                double[] inputData = input.GetData();
                Tensor[] inputSteps = new Tensor[contextWindowSize];
                for (int t = 0; t < contextWindowSize; t++)
                {
                    double[] stepData = new double[vocabSize];
                    Array.Copy(inputData, t * vocabSize, stepData, 0, vocabSize);
                    inputSteps[t] = new Tensor(stepData, new int[] { vocabSize });
                }

                Tensor[] h_ts = new Tensor[contextWindowSize];
                Tensor[] c_ts_all = new Tensor[contextWindowSize];
                Tensor[] c_prevs = new Tensor[contextWindowSize];
                Tensor[] i_ts = new Tensor[contextWindowSize];
                Tensor[] f_ts = new Tensor[contextWindowSize];
                Tensor[] c_tildes = new Tensor[contextWindowSize];
                Tensor[] o_ts = new Tensor[contextWindowSize];

                for (int t = 0; t < contextWindowSize; t++)
                {
                    Tensor h_prev =
                        (t == 0) ? new Tensor(new double[hiddenSize], new int[] { hiddenSize }) : h_ts[t - 1];
                    Tensor c_prev = (t == 0)
                        ? new Tensor(new double[hiddenSize], new int[] { hiddenSize })
                        : c_ts_all[t - 1];
                    c_prevs[t] = new Tensor(c_prev.GetData(), c_prev.GetShape());
                    (h_ts[t], c_ts_all[t], i_ts[t], f_ts[t], c_tildes[t], o_ts[t]) =
                        LSTMStep(inputSteps[t], h_prev, c_prev);
                }

                Tensor output = Forward(input);
                for (int o = 0; o < outputSize; o++)
                {
                    if (target.Infer(new int[] { o }) == 1.0)
                    {
                        epochLoss += -Math.Log(Math.Max(output.Infer(new int[] { o }), 1e-12));
                        break;
                    }
                }

                double[] grad_output_logits = new double[outputSize];
                for (int o = 0; o < outputSize; o++)
                {
                    grad_output_logits[o] = output.Infer(new int[] { o }) - target.Infer(new int[] { o });
                }

                for (int o = 0; o < outputSize; o++)
                {
                    for (int h = 0; h < hiddenSize; h++)
                    {
                        grad_W_out_data_acc[h * outputSize + o] += grad_output_logits[o] *
                                                                   h_ts[contextWindowSize - 1]
                                                                       .Infer(new int[] { h });
                    }

                    grad_b_out_data_acc[o] += grad_output_logits[o];
                }

                double[] grad_h_next = new double[hiddenSize];
                for (int h = 0; h < hiddenSize; h++)
                {
                    for (int o = 0; o < outputSize; o++)
                    {
                        grad_h_next[h] += grad_output_logits[o] * W_out.Infer(new int[] { h, o });
                    }
                }

                double[] grad_c_next = new double[hiddenSize];
                for (int t = contextWindowSize - 1; t >= 0; t--)
                {
                    Tensor h_prev_t =
                        (t == 0) ? new Tensor(new double[hiddenSize], new int[] { hiddenSize }) : h_ts[t - 1];
                    Tensor c_prev_t = c_prevs[t];
                    double[] grad_o_t = new double[hiddenSize];
                    for (int h = 0; h < hiddenSize; h++)
                        grad_o_t[h] = grad_h_next[h] * Tanh(c_ts_all[t].Infer(new int[] { h })) *
                                      SigmoidDerivative(o_ts[t].Infer(new int[] { h }));
                    double[] grad_c_t = new double[hiddenSize];
                    for (int h = 0; h < hiddenSize; h++)
                        grad_c_t[h] =
                            grad_h_next[h] * o_ts[t].Infer(new int[] { h }) *
                            TanhDerivative(Tanh(c_ts_all[t].Infer(new int[] { h }))) + grad_c_next[h];
                    double[] grad_c_tilde = new double[hiddenSize];
                    for (int h = 0; h < hiddenSize; h++)
                        grad_c_tilde[h] = grad_c_t[h] * i_ts[t].Infer(new int[] { h }) *
                                          TanhDerivative(c_tildes[t].Infer(new int[] { h }));
                    double[] grad_i_t = new double[hiddenSize];
                    for (int h = 0; h < hiddenSize; h++)
                        grad_i_t[h] = grad_c_t[h] * c_tildes[t].Infer(new int[] { h }) *
                                      SigmoidDerivative(i_ts[t].Infer(new int[] { h }));
                    double[] grad_f_t = new double[hiddenSize];
                    for (int h = 0; h < hiddenSize; h++)
                        grad_f_t[h] = grad_c_t[h] * c_prev_t.Infer(new int[] { h }) *
                                      SigmoidDerivative(f_ts[t].Infer(new int[] { h }));

                    double[] next_grad_h = new double[hiddenSize];
                    for (int h = 0; h < hiddenSize; h++)
                    {
                        for (int k = 0; k < hiddenSize; k++)
                        {
                            next_grad_h[h] += grad_i_t[k] * U_i.Infer(new int[] { h, k }) +
                                              grad_f_t[k] * U_f.Infer(new int[] { h, k }) +
                                              grad_c_tilde[k] * U_c.Infer(new int[] { h, k }) +
                                              grad_o_t[k] * U_o.Infer(new int[] { h, k });
                        }
                    }

                    grad_h_next = next_grad_h;
                    for (int h = 0; h < hiddenSize; h++) grad_c_next[h] = grad_c_t[h] * f_ts[t].Infer(new int[] { h });

                    for (int h = 0; h < hiddenSize; h++)
                    {
                        for (int k = 0; k < vocabSize; k++)
                        {
                            grad_W_i_acc[k * hiddenSize + h] += grad_i_t[h] * inputSteps[t].Infer(new int[] { k });
                            grad_W_f_acc[k * hiddenSize + h] += grad_f_t[h] * inputSteps[t].Infer(new int[] { k });
                            grad_W_c_acc[k * hiddenSize + h] += grad_c_tilde[h] * inputSteps[t].Infer(new int[] { k });
                            grad_W_o_acc[k * hiddenSize + h] += grad_o_t[h] * inputSteps[t].Infer(new int[] { k });
                        }

                        for (int k = 0; k < hiddenSize; k++)
                        {
                            grad_U_i_acc[k * hiddenSize + h] += grad_i_t[h] * h_prev_t.Infer(new int[] { k });
                            grad_U_f_acc[k * hiddenSize + h] += grad_f_t[h] * h_prev_t.Infer(new int[] { k });
                            grad_U_c_acc[k * hiddenSize + h] += grad_c_tilde[h] * h_prev_t.Infer(new int[] { k });
                            grad_U_o_acc[k * hiddenSize + h] += grad_o_t[h] * h_prev_t.Infer(new int[] { k });
                        }

                        grad_b_i_acc[h] += grad_i_t[h];
                        grad_b_f_acc[h] += grad_f_t[h];
                        grad_b_c_acc[h] += grad_c_tilde[h];
                        grad_b_o_acc[h] += grad_o_t[h];
                    }
                }
            }

            // --- MUDANÇA: Atualizar os pesos AQUI (depois de processar todo o mini-batch) ---

            grad_W_out_data_acc = ClipGradients(grad_W_out_data_acc);
            grad_b_out_data_acc = ClipGradients(grad_b_out_data_acc);
            grad_W_i_acc = ClipGradients(grad_W_i_acc);
            grad_U_i_acc = ClipGradients(grad_U_i_acc);
            grad_b_i_acc = ClipGradients(grad_b_i_acc);
            grad_W_f_acc = ClipGradients(grad_W_f_acc);
            grad_U_f_acc = ClipGradients(grad_U_f_acc);
            grad_b_f_acc = ClipGradients(grad_b_f_acc);
            grad_W_c_acc = ClipGradients(grad_W_c_acc);
            grad_U_c_acc = ClipGradients(grad_U_c_acc);
            grad_b_c_acc = ClipGradients(grad_b_c_acc);
            grad_W_o_acc = ClipGradients(grad_W_o_acc);
            grad_U_o_acc = ClipGradients(grad_U_o_acc);
            grad_b_o_acc = ClipGradients(grad_b_o_acc);

            UpdateWeights(W_out, grad_W_out_data_acc, _currentLearningRate);
            UpdateWeights(b_out, grad_b_out_data_acc, _currentLearningRate);
            UpdateWeights(W_i, grad_W_i_acc, _currentLearningRate);
            UpdateWeights(U_i, grad_U_i_acc, _currentLearningRate);
            UpdateWeights(b_i, grad_b_i_acc, _currentLearningRate);
            UpdateWeights(W_f, grad_W_f_acc, _currentLearningRate);
            UpdateWeights(U_f, grad_U_f_acc, _currentLearningRate);
            UpdateWeights(b_f, grad_b_f_acc, _currentLearningRate);
            UpdateWeights(W_c, grad_W_c_acc, _currentLearningRate);
            UpdateWeights(U_c, grad_U_c_acc, _currentLearningRate);
            UpdateWeights(b_c, grad_b_c_acc, _currentLearningRate);
            UpdateWeights(W_o, grad_W_o_acc, _currentLearningRate);
            UpdateWeights(U_o, grad_U_o_acc, _currentLearningRate);
            UpdateWeights(b_o, grad_b_o_acc, _currentLearningRate);

            return epochLoss / dataset.Count;
        }

        private void UpdateWeights(Tensor tensor, double[] grad, double learningRate)
        {
            double[] data = tensor.GetData();
            for (int i = 0; i < data.Length; i++)
            {
                data[i] -= learningRate * grad[i];
            }
        }

        private void CleanupOpenCL()
        {
            if (_matmulKernel.HasValue)
            {
                Cl.ReleaseKernel(_matmulKernel.Value);
                _matmulKernel = null;
            }

            if (_sigmoidKernel.HasValue)
            {
                Cl.ReleaseKernel(_sigmoidKernel.Value);
                _sigmoidKernel = null;
            }

            if (_tanhKernel.HasValue)
            {
                Cl.ReleaseKernel(_tanhKernel.Value);
                _tanhKernel = null;
            }

            if (_elementwiseAddKernel.HasValue)
            {
                Cl.ReleaseKernel(_elementwiseAddKernel.Value);
                _elementwiseAddKernel = null;
            }

            if (_elementwiseAddBroadcastKernel.HasValue)
            {
                Cl.ReleaseKernel(_elementwiseAddBroadcastKernel.Value);
                _elementwiseAddBroadcastKernel = null;
            }

            if (_elementwiseMultiplyKernel.HasValue)
            {
                Cl.ReleaseKernel(_elementwiseMultiplyKernel.Value);
                _elementwiseMultiplyKernel = null;
            }

            if (_program.HasValue)
            {
                Cl.ReleaseProgram(_program.Value);
                _program = null;
            }

            if (_queue.HasValue)
            {
                Cl.ReleaseCommandQueue(_queue.Value);
                _queue = null;
            }

            if (_context.HasValue)
            {
                Cl.ReleaseContext(_context.Value);
                _context = null;
            }

            _device = null;
        }

        public void Dispose()
        {
            CleanupOpenCL();
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

        private double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-Math.Max(-500, Math.Min(500, x))));
        private double Tanh(double x) => Math.Tanh(Math.Max(-500, Math.Min(500, x)));
        private double SigmoidDerivative(double sigmoidOutput) => sigmoidOutput * (1 - sigmoidOutput);
        private double TanhDerivative(double tanhOutput) => 1 - tanhOutput * tanhOutput;

        public void SaveModel(string filePath)
        {
            try
            {
                var modelData = new NeuralNetworkModelData
                {
                    InputSize = inputSize,
                    HiddenSize = hiddenSize,
                    OutputSize = outputSize,
                    ContextWindowSize = contextWindowSize,
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
                    { WriteIndented = true, NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals };
                string jsonString = JsonSerializer.Serialize(modelData, options);
                File.WriteAllText(filePath, jsonString);
                Console.WriteLine($"Modelo salvo com sucesso em: {filePath}");
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
                if (!File.Exists(filePath)) return null;
                string jsonString = File.ReadAllText(filePath);
                var modelData = JsonSerializer.Deserialize<NeuralNetworkModelData>(jsonString);
                if (modelData == null) throw new Exception("Dados do modelo JSON estão nulos.");

                int inferredContextWindowSize = modelData.ContextWindowSize > 0
                    ? modelData.ContextWindowSize
                    : (modelData.OutputSize > 0)
                        ? modelData.InputSize / modelData.OutputSize
                        : 0;

                if (inferredContextWindowSize == 0)
                    throw new Exception("Não foi possível inferir ContextWindowSize do modelo salvo.");

                Console.WriteLine($"Modelo carregado com sucesso de: {filePath}");

                return new NeuralNetwork(modelData.InputSize, modelData.HiddenSize, modelData.OutputSize,
                    inferredContextWindowSize,
                    new Tensor(modelData.WeightsInputGate.data, modelData.WeightsInputGate.shape),
                    new Tensor(modelData.RecurrentWeightsInputGate.data, modelData.RecurrentWeightsInputGate.shape),
                    new Tensor(modelData.BiasInputGate.data, modelData.BiasInputGate.shape),
                    new Tensor(modelData.WeightsForgetGate.data, modelData.WeightsForgetGate.shape),
                    new Tensor(modelData.RecurrentWeightsForgetGate.data, modelData.RecurrentWeightsForgetGate.shape),
                    new Tensor(modelData.BiasForgetGate.data, modelData.BiasForgetGate.shape),
                    new Tensor(modelData.WeightsCellGate.data, modelData.WeightsCellGate.shape),
                    new Tensor(modelData.RecurrentWeightsCellGate.data, modelData.RecurrentWeightsCellGate.shape),
                    new Tensor(modelData.BiasCellGate.data, modelData.BiasCellGate.shape),
                    new Tensor(modelData.WeightsOutputGate.data, modelData.WeightsOutputGate.shape),
                    new Tensor(modelData.RecurrentWeightsOutputGate.data, modelData.RecurrentWeightsOutputGate.shape),
                    new Tensor(modelData.BiasOutputGate.data, modelData.BiasOutputGate.shape),
                    new Tensor(modelData.WeightsOutput.data, modelData.WeightsOutput.shape),
                    new Tensor(modelData.BiasOutput.data, modelData.BiasOutput.shape));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erro ao carregar o modelo: {ex.Message}");
                return null;
            }
        }
    }
}