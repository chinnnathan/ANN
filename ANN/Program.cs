﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using ANN.Utils;
using ANN.Networks;
using ANN.ANNTemplates;

namespace ANN
{
    class Program
    {
        private string _networkType = "Backpropogation";
        private string _trainingFile = "D:/Science/ANN/LetterTrainingHW1.dat";
        private string _testingFile = "D:/Science/ANN/LetterTestingHW1.dat";
        private string _validationFile = "D:/Science/ANN/Validation.csv";
        private string _outputTestingFile = "TestingOutput.csv";
        private string _outputTrainingFile = "TrainingOutput.csv";
        private double _learningRate = 0.005;
        private int    _hiddenLayers = 1;

        public string NetworkType { get { return _networkType; } set { _networkType=value; } }
        public string TestingFile { get { return _testingFile; } set { _testingFile=value; } }
        public string TrainingFile { get { return _trainingFile; } set { _trainingFile=value; } }
        public string ValidationFile { get { return _validationFile; } set { _validationFile=value; } }
        public string OutputTestingFile { get { return _outputTestingFile; } set { _outputTestingFile=value; } }
        public string OutputTrainingFile { get { return _outputTrainingFile; } set { _outputTrainingFile=value; } }
        public double LearningRate { get { return _learningRate; } set { _learningRate=value; } }
        public int HiddenLayers { get { return _hiddenLayers; } set { _hiddenLayers=value; } }

        static void Main(string[] args)
        {
            Program pr = new Program();
            string ui = "r";
            do
            {
                pr.RunProgram(args);
                Console.WriteLine("Enter q to quit: ");
                ui = Console.ReadLine();
                if (ui == "q")
                    break;
                else
                    args = ui.Split(' ').ToList().ToArray();
            } while (ui != "q");
        }

        public void RunProgram(string[] args)
        {
            SetVariables(args);
        }

        public void SetVariables(string[] args)
        {
            for (int i=0; i<args.Length; i+=2)
            {
                string var = args[i].Trim('-');
                string value = args[i + 1];
                this.SetValueByString(var, value);
            }
        }
    }
}
