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
    }
}
