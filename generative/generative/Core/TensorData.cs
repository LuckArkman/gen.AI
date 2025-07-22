using System;
using System.IO;
using Random = System.Random; 
using System.Text.Json; 
using System.Text.Json.Serialization; 

namespace Core
{
    public class TensorData
    {
        public double[]? data { get; set; }
        public int[]? shape { get; set; }
    }
}