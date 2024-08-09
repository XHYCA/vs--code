print('-'*20,'欢迎使用员工管理系统', '-'*20)

emps = ['jack\t18\t男\t未知']

while True:
    print('请选择操作：')
    print('\t1.查询员工信息')
    print('\t2.添加员工信息')
    print('\t3.删除员工信息')
    print('\t4.退出系统')
    u = input('请选择1-4:')
    print('-'*62)
    if u == '1':
        print( '序号\t姓名\t年龄\t性别\t住址')
        n = 1
        for emp in emps:
            print(f'{n}\t{emp}')
            n += 1
    elif u == '2':
        emp_n = input('员工姓名:')
        emp_a = input('员工年龄:')
        emp_g = input('员工性别:')
        emp_z = input('员工住址:')
        emp = f'{emp_n}\t{emp_a}\t{emp_g}\t{emp_z}'
        uc = input(f'是否添加\n{emp}?(y/n)')
        print( '序号\t姓名\t年龄\t性别\t住址')
        if uc == 'y':
            emps.append(emp)
            for emp in emps:
                n = 1
                print(f'{n}\t{emp}')
                n += 1
            print('添加成功！')
        else:
            pass
            print('取消成功！')
    elif u == '3':
        choice = input('请选择要删除的员工序号:')
        if 1 <= int(choice) <= len(emps):
            del emps[ int(choice)-1 ]
        else:
            print('序号错误，请重新输入')
    elif u == '4':
        print('谢谢使用！')
        break
    else:
        print('输入错误，请重新输入')
    print('-'*62)