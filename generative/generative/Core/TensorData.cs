using System;
using System.IO;
using Random = System.Random; 
using System.Text.Json; 
using System.Text.Json.Serialization; 

namespace Core
{
    public class TensorData
    {
        public double[] data { get; set; } = Array.Empty<double>(); // Inicializado para evitar null
        public int[] shape { get; set; } = Array.Empty<int>(); // Inicializado para evitar null
    }
}