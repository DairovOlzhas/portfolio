using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace SnakeGame
{
    public class Snake
    {
        public List<Point> body;
        public char sign = '*';
        public ConsoleColor color = ConsoleColor.Yellow;

        public Snake() { }
        public Snake(int a)
        {
            body = new List<Point>();
            body.Add(new Point(39, 15));
            body.Add(new Point(38, 15));
            body.Add(new Point(37, 15));
            body.Add(new Point(36, 15));
            body.Add(new Point(35, 15));
        }

        public void Move(int dx, int dy)
        {
            for (int i = body.Count - 1; i > 0; i--)
            {
                body[i].x = body[i - 1].x;
                body[i].y = body[i - 1].y;
            }

            body[0].x += dx;
            body[0].y += dy;

            if (body[0].x > Game.sizex - 1)
                body[0].x = 0;
            if (body[0].x < 0)
                body[0].x = Game.sizex - 1;
            if (body[0].y > Game.sizey - 1)
                body[0].y = 2;
            if (body[0].y < 2)
                body[0].y = Game.sizey - 1;
        }
        
        

        public void Draw()
        {
            int i = 0;
            while (i < body.Count)
            {
                Console.SetCursorPosition(body[i].x, body[i].y);
                Console.ForegroundColor = (i == 0) ? ConsoleColor.Red : color;
                char signs = (i == body.Count - 1) ? ' ' : sign;
                Console.Write(signs);
                i++;
            }
        }
    }
}
