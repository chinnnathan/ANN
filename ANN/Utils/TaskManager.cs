using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANN.Utils 
{
    public class TaskManager : IDisposable
    {
        public int CurrentTaskQueueCount { get { return _Queue.Count; } }
        public dynamic LastResult;

        private bool _Run = true;
        BlockingCollection<Tuple<Delegate,object[]>> _Queue = new BlockingCollection<Tuple<Delegate,object[]>>();

        public TaskManager()
        {
            Task.Factory.StartNew(()=>
            {
                while(_Run)
                {
                    try
                    {
                        var action = _Queue.Take();
                        action.Item1.DynamicInvoke(action.Item2);
                    }
                    catch (InvalidOperationException)
                    {
                        _Run = false;
                        Debug.WriteLine("TaskManager finished tasks");
                    }
                    catch
                    {
                        Debug.WriteLine("Unexpected error found");
                    }
                }
            });
        }

        public void Append(Action del, params object[] args)
        {
            _Queue.Add(new Tuple<Delegate, object[]>(del, args));
        }

        public void Dispose()
        {
            _Queue.CompleteAdding();
        }
    }
}
