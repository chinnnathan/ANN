using System;

using ANN.Utils;
using System.Collections.Generic;

namespace RecurrentNeural
{
  class RecurrentNeuralProgram
  {
        /*static void Main(string[] args)
        {
          Console.WriteLine("\nBegin recurrent neural network demo\n");

          int numInput = 2;
          int numHidden = 3;
          int numOutput = 2;
          int seed = 0;

          Console.WriteLine("Creating a 2-3-2 recurrent neural network");
          RecurrentNetwork rn = new RecurrentNetwork(numInput, numHidden, numOutput, seed);

          double[] wts = new double[] { 0.01, 0.02, 0.03, 0.04, 0.05, 0.06,
          0.07, 0.08, 0.09,
          0.10, 0.11, 0.12, 0.13, 0.14, 0.15,
          0.16, 0.17,
          0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26 };
          Console.WriteLine("\nSetting weights and biases to 0.01 to 0.26");
          rn.SetWeights(wts);

          double[] cValues = new double[] { 0.3, 0.4, 0.5 };
          Console.WriteLine("Setting context nodes to 0.3, 0.4, 0.5");
          rn.SetContext(cValues);

          double[] xValues = new double[] { 1.0, 2.0 };
          Console.WriteLine("Setting input nodes to 1.0, 2.0");

          Console.WriteLine("\nComputing outputs\n");
          double[] yValues = rn.ComputeOutputs(xValues);
          Console.WriteLine("\nFinished computing");

          Console.WriteLine("\nOutput node values are:");
          ShowVector(yValues, 4, yValues.Length, true);

          Console.WriteLine("\nEnd recurrent neural network demo\n");
          Console.ReadLine();
        } // Main

        // -----------------------------------------
        //*/

        public int numInput, numHidden, numOutput, seed;

        public void RunRNN(string inputfile, string correctfile, int cluster = 1, int[] ignore = null)
        {
            List<List<double>> inputs = FileManager.ReadFileToDouble(inputfile, cluster, ignore, normalize:false);
            RecurrentNetwork rn = new RecurrentNetwork(numInput, numHidden, numOutput, seed);
            
            // Weights for Input->Hidden + Hidden->Context + Context->Hidden + Hidden->Output + Bias
            int numWeights = (numInput * numHidden) + (numHidden) + (numHidden * numHidden) + (numHidden * numOutput) + (numOutput);

            double[] weights = AdvancedMath.GetRandomRanges(numWeights, -1.0, 1.0);
            rn.SetWeights(weights);

            double[] cVal = AdvancedMath.GetRandomRanges(numHidden, -1.0, 1.0);
            rn.SetContext(cVal);

            double[] iVal = AdvancedMath.GetRandomRanges(numInput, -1.0, 1.0);
            double[] yVal = rn.ComputeOutputs(iVal);
        }


    public static void ShowMatrix(double[][] matrix, int numRows,
      int decimals, bool indices)
    {
      int len = matrix.Length.ToString().Length;
      for (int i = 0; i < numRows; ++i)
      {
        if (indices == true)
          Console.Write("[" + i.ToString().PadLeft(len) + "]  ");
        for (int j = 0; j < matrix[i].Length; ++j)
        {
          double v = matrix[i][j];
          if (v >= 0.0)
            Console.Write(" "); // '+'
          Console.Write(v.ToString("F" + decimals) + "  ");
        }
        Console.WriteLine("");
      }

      if (numRows < matrix.Length)
      {
        Console.WriteLine(". . .");
        int lastRow = matrix.Length - 1;
        if (indices == true)
          Console.Write("[" + lastRow.ToString().PadLeft(len) + "]  ");
        for (int j = 0; j < matrix[lastRow].Length; ++j)
        {
          double v = matrix[lastRow][j];
          if (v >= 0.0)
            Console.Write(" "); // '+'
          Console.Write(v.ToString("F" + decimals) + "  ");
        }
      }
      Console.WriteLine("\n");
    }

    public static void ShowVector(double[] vector, int decimals,
      int lineLen, bool newLine)
    {
      for (int i = 0; i < vector.Length; ++i)
      {
        if (i > 0 && i % lineLen == 0) Console.WriteLine("");
        if (vector[i] >= 0) Console.Write(" ");
        Console.Write(vector[i].ToString("F" + decimals) + " ");
      }
      if (newLine == true)
        Console.WriteLine("");
    }

    // -----------------------------------------

  } // Program

  public class RecurrentNetwork
  {
    private int numInput; // number input nodes
    private int numHidden;
    private int numOutput;
    // private int numContext; // = numHidden

    private double[] inputs;

    private double[][] ihWeights; // input-hidden
    private double[] hBiases;
    private double[] hNodes; // hidden nodes

    private double[][] chWeights; // context-hidden
    private double[] cNodes; // context nodes

    private double[][] hoWeights; // hidden-output
    private double[] oBiases;

    private double[] outputs;
    private Random rnd;

    public RecurrentNetwork(int numInput, int numHidden,
      int numOutput, int seed)
    {
      this.numInput = numInput;
      this.numHidden = numHidden;
      this.numOutput = numOutput;

      this.inputs = new double[numInput];

      this.ihWeights = MakeMatrix(numInput, numHidden);
      this.hBiases = new double[numHidden];
      this.hNodes = new double[numHidden];

      this.chWeights = MakeMatrix(numHidden, numHidden);
      this.cNodes = new double[numHidden];

      this.hoWeights = MakeMatrix(numHidden, numOutput);
      this.oBiases = new double[numOutput];
      this.outputs = new double[numOutput];

      this.rnd = new Random(seed);
      this.InitializeWeights(); // weights and biases
      this.InitializeContext(); // context nodes
    } // ctor

    private static double[][] MakeMatrix(int rows,
      int cols) // helper for ctor, Train
    {
      double[][] result = new double[rows][];
      for (int r = 0; r < result.Length; ++r)
        result[r] = new double[cols];
      return result;
    }

    private void InitializeWeights() // helper for ctor
    {
      // initialize weights and biases to random values between 0.0001 and 0.001
      int numWeights = (numInput * numHidden) +
        (numHidden * numOutput) +
        (numHidden * numHidden) + numHidden + numOutput;
      double[] initialWeights = new double[numWeights];
      for (int i = 0; i < initialWeights.Length; ++i)
        initialWeights[i] = (0.001 - 0.0001) * rnd.NextDouble() + 0.0001;
      
      this.SetWeights(initialWeights);
    } // InitializeWeights

    private void InitializeContext()
    {
      for (int c = 0; c < numHidden; ++c)
        cNodes[c] = 0.3 + (0.1 * c); // .3, .4, .5
    }

    public void SetWeights(double[] weights)
    {
      // copy serialized weights and biases in weights[] array
      // to i-h weights, h biases, h-o weights, o biases, c-h weights
      int numWeights = (numInput * numHidden) +
        (numHidden * numOutput) +
        (numHidden * numHidden) + numHidden + numOutput;
      if (weights.Length != numWeights)
        throw new Exception("Bad weights array in SetWeights");

      int p = 0; // points into weights param

      for (int i = 0; i < numInput; ++i) // ih
        for (int j = 0; j < numHidden; ++j)
          ihWeights[i][j] = weights[p++];
      for (int i = 0; i < numHidden; ++i) // h bias
        hBiases[i] = weights[p++];
      for (int j = 0; j < numHidden; ++j) // ho
        for (int k = 0; k < numOutput; ++k)
          hoWeights[j][k] = weights[p++];
      for (int k = 0; k < numOutput; ++k) // o bias
        oBiases[k] = weights[p++];
      for (int c = 0; c < numHidden; ++c) // ch
        for (int j = 0; j < numHidden; ++j)
          chWeights[c][j] = weights[p++];
    } // SetWeights

    public double[] GetWeights()
    {
      int numWeights = (numInput * numHidden) +
        (numHidden * numOutput) +
        (numHidden * numHidden) + numHidden + numOutput;

      double[] result = new double[numWeights];
      int k = 0;
      for (int i = 0; i < ihWeights.Length; ++i)
        for (int j = 0; j < ihWeights[0].Length; ++j)
          result[k++] = ihWeights[i][j];
      for (int i = 0; i < hBiases.Length; ++i)
        result[k++] = hBiases[i];
      for (int i = 0; i < hoWeights.Length; ++i)
        for (int j = 0; j < hoWeights[0].Length; ++j)
          result[k++] = hoWeights[i][j];
      for (int i = 0; i < oBiases.Length; ++i)
        result[k++] = oBiases[i];
      return result;
    } // GetWeights

    public void SetContext(double[] values)
    {
      if (values.Length != this.numHidden)
        throw new Exception("Bad array in SetContext");
      for (int c = 0; c < numHidden; ++c)
        cNodes[c] = values[c];
    }

    public double[] ComputeOutputs(double[] xValues)
    {
      double[] hSums = new double[numHidden]; // hidden nodes sums scratch array
      double[] oSums = new double[numOutput]; // output nodes sums

      for (int i = 0; i < xValues.Length; ++i) // copy x-values to inputs
        this.inputs[i] = xValues[i];

      for (int j = 0; j < numHidden; ++j)  // compute i-h sum of weights * inputs
        for (int i = 0; i < numInput; ++i)
          hSums[j] += this.inputs[i] * this.ihWeights[i][j]; // note +=

      // add in context nodes * c-h weights
      for (int j = 0; j < numHidden; ++j)
        for (int c = 0; c < numHidden; ++c) // all c nodes contribute to each h node
          hSums[j] += this.cNodes[c] * chWeights[c][j]; // note +=

      for (int j = 0; j < numHidden; ++j)  // add biases to input-to-hidden sums
        hSums[j] += this.hBiases[j];

      Console.WriteLine("Pre-activation hNode values:");
      RecurrentNeuralProgram.ShowVector(hSums, 4, hSums.Length, true);

      for (int j = 0; j < numHidden; ++j)   // apply activation
        this.hNodes[j] = HyperTan(hSums[j]); // hard-coded

      Console.WriteLine("Post-activation hNode values:");
      RecurrentNeuralProgram.ShowVector(hNodes, 4, hNodes.Length, true);

      for (int k = 0; k < numOutput; ++k)   // compute h-o sum of weights * hOutputs
        for (int j = 0; j < numHidden; ++j)
          oSums[k] += hNodes[j] * hoWeights[j][k];

      for (int k = 0; k < numOutput; ++k)  // add biases to input-to-hidden sums
        oSums[k] += oBiases[k];

      double[] softOut = Softmax(oSums); // all outputs at once for efficiency
      for (int k = 0; k < numOutput; ++k)
        outputs[k] = softOut[k];

      // copy h node value to corresponding c node
      for (int j = 0; j < numHidden; ++j)
        cNodes[j] = hNodes[j];
      Console.WriteLine("New cNode values:");
      RecurrentNeuralProgram.ShowVector(cNodes, 4, cNodes.Length, true);

      double[] retResult = new double[numOutput]; // could define a GetOutputs 
      for (int k = 0; k < numOutput; ++k)
        retResult[k] = outputs[k];
      return retResult;
    } // ComputeOutputs

    private static double HyperTan(double x)
    {
      if (x < -20.0) return -1.0; // approximation is correct to 30 decimals
      else if (x > 20.0) return 1.0;
      else return Math.Tanh(x);
    }

    private static double[] Softmax(double[] oSums)
    {
      // does all output nodes at once so scale
      // doesn't have to be re-computed each time
      double[] result = new double[oSums.Length];

      double sum = 0.0;
      for (int k = 0; k < oSums.Length; ++k)
        sum += Math.Exp(oSums[k]);

      for (int k = 0; k < oSums.Length; ++k)
        result[k] = Math.Exp(oSums[k]) / sum;

      return result; // now scaled so that xi sum to 1.0
    }

  } // NeuralNetwork

} // ns
