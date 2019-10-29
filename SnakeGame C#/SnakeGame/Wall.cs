using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SnakeGame
{
    public class Wall
    {
        public List<Point> body;
        public char sign = '#';
        public ConsoleColor color = ConsoleColor.Magenta;

        public Wall()
        {
            body = new List<Point>();
            ReadWallLevel(Game.level);
        }

        public void ReadWallLevel(int level)
        {
            body.Clear();
            string[] walls = File.ReadAllLines(string.Format(@"Levels/Level{0}.txt", level));
            Game.score = int.Parse(walls[0]);
            for(int i = 2; i < walls.Length; i++)
                for(int j = 0; j < walls[i].Length; j++)
                    if (walls[i][j] == '#')
                        body.Add(new Point(j, i));
                
        }

        public void Draw()
        {
            
            foreach (Point p in body)
            {
                Console.ForegroundColor = color;
                Console.SetCursorPosition(p.x, p.y);
                Console.Write(sign);                
            }
        }
    }
}
