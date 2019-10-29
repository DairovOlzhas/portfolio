import { Injectable, EventEmitter } from '@angular/core';
import { MainService } from './main.service';
import { HttpClient } from '@angular/common/http';
import { TaskList, Task, IAuthResponse } from '../models/models';

@Injectable({
  providedIn: 'root'
})
export class ApiService extends MainService {


  constructor(http: HttpClient) {
    super(http);
  }

  getTaskLists(): Promise<TaskList[]> {
    // console.log(this.get(`http://127.0.0.1:8000/api/task_lists/`, {}));
    return this.get(`http://127.0.0.1:8000/api/task_lists/`, {});
  }

  createTaskList(nme: any): Promise<TaskList> {
    // console.log(this.get(`http://127.0.0.1:8000/api/task_lists/`, {}));
    return this.post(`http://127.0.0.1:8000/api/task_lists/`, {
      name: nme
    });
  }
  getTaskList(tasklist: TaskList): Promise<TaskList> {
    return this.get(`http://127.0.0.1:8000/api/task_lists/${tasklist.id}/`, {});
  }
  updateTaskList(tasklist: TaskList): Promise<TaskList> {
    return this.put(`http://127.0.0.1:8000/api/task_lists/${tasklist.id}/`, {
      name: tasklist.name
    });
  }
  deleteTaskList(tasklist: TaskList): Promise<any> {
    return this.delet(`http://127.0.0.1:8000/api/task_lists/${tasklist.id}/`, {});
  }
  getTaskListsTask(tasklist: TaskList): Promise<Task[]> {
    return this.get(`http://127.0.0.1:8000/api/task_lists/${tasklist.id}/tasks/`, {});
  }
  getTasks(): Promise<Task[]> {
    return this.get(`http://127.0.0.1:8000/api/tasks/`, {});
  }
  createTask(nme: any, createdAt: any, dueOn: any, sttus: any, tasklist: number): Promise<Task> {
    return this.post(`http://127.0.0.1:8000/api/task_lists/${tasklist}/tasks/`, {
      name: nme,
      created_at: createdAt,
      due_on: dueOn,
      status: sttus,
      task_list_id: tasklist
    });
  }
  getTask(task: Task): Promise<Task> {
    return this.get(`http://127.0.0.1:8000/api/tasks/${task.id}/`, {});
  }
  updateTask(task: Task): Promise<any> {
    return this.put(`http://127.0.0.1:8000/api/tasks/${task.id}/`, {
      name: task.name,
      created_at: task.created_at,
      due_on: task.due_on,
      status: task.status,
      task_list_id: task.task_list_id
    });
  }
  deleteTask(task: Task): Promise<any> {
    return this.delet(`http://127.0.0.1:8000/api/tasks/${task.id}/`, {});
  }

  auth(login: any, pasword: any): Promise<IAuthResponse> {
    return this.post('http://127.0.0.1:8000/api/login/', {
      username: login,
      password: pasword
    });
  }


  logout(): Promise<any> {
    return this.post('http://127.0.0.1:8000/api/logout/', {});
  }

  search(query: string): Promise<Task[]> {
    return this.get(`http://127.0.0.1:8000/api/task_lists/3/tasks/?search=${query}`, {});
  }

  FiltrCreatedDate(max: number, min: number): Promise<Task[]> {
    return this.get(`http://127.0.0.1:8000/api/task_lists/3/tasks/?maxlikecount=${max}&minlikecount=${min}`, {});
  }



  OrderBy(value: any): Promise<Task[]> {
    return this.get(`http://127.0.0.1:8000/api/task_lists/3/tasks/?ordering=${value}`, {});
  }
}
