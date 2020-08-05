def solution(expression):
    answer = 0
    import itertools

    ch = ['*','-','+']
    lst = [] #숫자 담을 리스트
    op = [] #연산자 기호 담을 리스트
    
    for t in expression: #연산자 담음
        if t == '+' or t == '-' or t =='*':
            op.append(t)
    for t in ch: #연산자 지움
        expression = expression.replace(t,' ')

    lst = list(map(int,expression.split())) #숫자 담음

 
    for gr in list(map(''.join, itertools.permutations(ch))): #기호의 순열 만들어줌
        op_t = list(op) #매 연산에 새로 사용해야하므로 따로 만들어줌
        lst_t = list(lst)

        for g in list(gr) : #현재 순서대로 기호 넣어줌
            while g in op_t: 
                tmp =lst_t[op_t.index(g)]
                if g == '*':
                    tmp *= lst_t[op_t.index(g)+1]  
                elif g == '-':
                    tmp -= lst_t[op_t.index(g)+1]  
                elif g == '+':
                    tmp += lst_t[op_t.index(g)+1]  
                else:
                    break 
                del lst_t[int(op_t.index(g))] #사용한 숫자 지워준다 
                lst_t[op_t.index(g)] = tmp #(덮어써서 다음 연산에 쓸 것 하나는 남겨둔다)
                del op_t[op_t.index(g)]  #사용한 연산자도 지워준다
        if abs(lst_t[0]) > answer : #결과 값의 절대값이 더 크면 덮어쓴다
            answer = abs(lst_t[0])  
    return answer