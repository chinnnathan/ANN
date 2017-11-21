using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Reflection;

namespace ANN.Utils
{
    public static class OperationCLI
    {
        public static void SetValueByString(this object obj, string str, string setVal)
        {
            try
            {
                PropertyInfo propertyInfo = obj.GetType().GetProperty(str);
                if (propertyInfo.PropertyType.IsArray)
                {
                    var array = Array.CreateInstance(propertyInfo.PropertyType.GetElementType(), setVal.Count(f => f ==',') + 1);
                    var values = setVal.Split(',').Select(n => Convert.ToDouble(n)).ToArray();
                    for (int i = 0; i < array.Length; i++)
                        array.SetValue(Convert.ChangeType(values[i], propertyInfo.PropertyType.GetElementType()), i);
                    propertyInfo.SetValue(obj, array);
                }

                else
                    propertyInfo.SetValue(obj, Convert.ChangeType(setVal, propertyInfo.PropertyType), null);
            }
            catch
            {
                Console.WriteLine("Cannot Find Variable: {0}", str);
            }
        }

        public static List<Tuple<string, dynamic>> GetVariableNames(this object obj)
        {
            List<string> names = new List<string>();
            List<dynamic> vals = new List<dynamic>();
            List<Tuple<string, dynamic>> ret = new List<Tuple<string, dynamic>>();
            try
            {
                names = obj.GetType().GetProperties().Select(x => x.Name).ToList();
                vals = obj.GetType().GetProperties().Select(x => x.GetValue(obj)).ToList();
                var zipObj = names.Zip(vals, (n, v) => new { Name = n, Value = v });
                foreach(var z in zipObj)
                {
                    ret.Add(new Tuple<string, dynamic>(z.Name, z.Value));
                }
            }
            catch
            {
                Console.WriteLine("Cannot access object");  
            }
            return ret;
        }
    }
}
