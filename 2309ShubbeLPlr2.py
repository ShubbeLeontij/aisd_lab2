#!/usr/bin/env python3
import math
import time
import random
from enum import Enum
import json


def bubble_sort(array):
    n = len(array)
    for i in range(n):
        already_sorted = True
        for j in range(n - i - 1):
            if array[j] > array[j + 1]:
                array[j], array[j + 1] = array[j + 1], array[j]
                already_sorted = False
        if already_sorted:
            break
    return array


def selection_sort(array):
    for i in range(len(array)):
        min_index = i
        for j in range(i + 1, len(array)):
            if array[j] < array[min_index]:
                min_index = j
        (array[i], array[min_index]) = (array[min_index], array[i])
    return array


def insertion_sort(array):
    for i in range(1, len(array)):
        key_item = array[i]
        j = i - 1
        while j >= 0 and array[j] > key_item:
            array[j + 1] = array[j]
            j -= 1
        array[j + 1] = key_item
    return array


def shell_sort(array, mode=0):
    def get_first(N, mode=0):
        if mode == 0:  # Shell
            return 0, N // 2
        if mode == 1:  # Knuth
            iterator = 1
            while (3**iterator - 1) // 2 < N // 3:
                iterator += 1
            step = (3**iterator - 1) // 2
            return iterator - 1, step
        if mode == 2:  # Hibbard
            iterator = 1
            while 2**iterator - 1 < N:
                iterator += 1
            step = 2**(iterator - 1) - 1
            return iterator - 1, step

    def get_next(gap, iterator, mode=0):
        if mode == 0:  # Shell
            return 0, gap // 2
        if mode == 1:  # Knuth
            step = (3**iterator - 1) // 2
            return iterator - 1, step
        if mode == 2:  # Hibbard
            step = 2**(iterator - 1) - 1
            return iterator - 1, step

    N = len(array)
    iterator, gap = get_first(N, mode)
    while gap > 0:
        for i in range(gap, N):
            temp = array[i]
            j = i
            while j >= gap and array[j - gap] > temp:
                array[j] = array[j - gap]
                j -= gap
            array[j] = temp
        iterator, gap = get_next(gap, iterator, mode)
    return array


def quick_sort(array):
    if len(array) <= 1:
        return array
    else:
        left = array[0]
        mid = array[len(array) // 2]
        right = array[-1]
        if left >= right:
            if right >= mid:
                sup = right
            elif left >= mid:
                sup = mid
            else:
                sup = left
        else:
            if right <= mid:
                sup = right
            elif left <= mid:
                sup = mid
            else:
                sup = left
        left_array = []
        right_array = []
        mid_array = []
        for el in array:
            if el < sup:
                left_array.append(el)
            elif el > sup:
                right_array.append(el)
            else:
                mid_array.append(el)
        return quick_sort(left_array) + mid_array + quick_sort(right_array)


def merge_sort(array):
    def merge(left, right):
        N = len(left)
        K = len(right)
        if N == 0:
            return right
        if K == 0:
            return left
        array = []
        i = j = 0
        while i < N and j < K:
            if left[i] <= right[j]:
                array.append(left[i])
                i += 1
            else:
                array.append(right[j])
                j += 1
        if j == K:
            array += left[i:]
        if i == N:
            array += right[j:]
        return array

    if len(array) < 2:
        return array
    mid = len(array) // 2
    return merge(left=merge_sort(array[:mid]), right=merge_sort(array[mid:]))


def heap_sort(array):
    def heapify(array, n, i):
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2
        if l < n and array[i] < array[l]:
            largest = l
        if r < n and array[largest] < array[r]:
            largest = r
        if largest != i:
            (array[i], array[largest]) = (array[largest], array[i])
            heapify(array, n, largest)
        return array

    n = len(array)
    for i in range(n // 2 - 1, -1, -1):
        array = heapify(array, n, i)
    for i in range(n - 1, 0, -1):
        (array[i], array[0]) = (array[0], array[i])
        heapify(array, i, 0)
    return array


def tim_sort(array):
    def find_min_run(N):
        R = 0
        while N >= TIM_SORT_MIN_RUN:
            R |= N & 1
            N >>= 1
        return N + R

    def binary_search(array, left, right, x):
        while left <= right:
            mid = (right + left) // 2
            if array[mid] == x:
                return mid
            elif array[mid] < x:
                left = mid + 1
            else:
                right = mid - 1
        return left

    def merge(left, right):
        N = len(left)
        K = len(right)
        if N == 0:
            return right
        if K == 0:
            return left
        array = []
        i = j = 0
        galop_l = galop_r = TIM_SORT_GALOP
        while i < N and j < K:
            if left[i] <= right[j]:
                if galop_l >= TIM_SORT_GALOP:
                    index = binary_search(left, i, N - 1, right[j])
                    array += left[i:index]
                    i = index
                    galop_l = 0
                    continue
                array.append(left[i])
                i += 1
                galop_l += 1
                galop_r = 0
            else:
                if galop_r >= TIM_SORT_GALOP:
                    index = binary_search(right, j, K - 1, left[i])
                    array += right[j:index]
                    j = index
                    galop_r = 0
                    continue
                array.append(right[j])
                j += 1
                galop_r += 1
                galop_l = 0
        if j == K:
            array += left[i:]
        if i == N:
            array += right[j:]
        return array

    def check(runs):
        while len(runs) > 2:
            run1 = runs.pop()
            run2 = runs.pop()
            run3 = runs.pop()
            if len(run1) > len(run2) + len(run3):
                if len(run2) < len(run3):
                    runs.append(run3)
                    runs.append(merge(run1, run2))
                else:
                    runs.append(run3)
                    runs.append(run2)
                    runs.append(run1)
                    break
            else:
                if len(run1) > len(run3):
                    runs.append(merge(run2, run3))
                    runs.append(run1)
                else:
                    runs.append(run3)
                    runs.append(merge(run1, run2))
        return runs

    N = len(array)
    if N <= TIM_SORT_MIN_RUN:
        return insertion_sort(array)
    min_run = find_min_run(N)
    runs_stack = []
    start = index = 0
    cntr = 0
    increasing = None
    while index < N - 1:
        if array[index] == array[index + 1]:
            index += 1
            cntr += 1
            continue
        if array[index] < array[index + 1]:
            if increasing is None:
                increasing = True
                index += 1
                continue
            if increasing is True:
                index += 1
                continue
            index = min(max(index, start + min_run), N)
            run = insertion_sort(array[start:index])
        else:
            if increasing is None:
                increasing = False
                index += 1
                continue
            if increasing is False:
                index += 1
                continue
            index = min(max(index, start + min_run), N)
            run = insertion_sort(array[start:index][::-1])
        start = index
        increasing = None
        runs_stack.append(run)
        runs_stack = check(runs_stack)
    if increasing is True:
        runs_stack.append(insertion_sort(array[start:]))
    if increasing is False:
        runs_stack.append(insertion_sort(array[start:][::-1]))
    while i := len(runs_stack) > 1:
        if i > 2 and len(runs_stack[-3]) < len(runs_stack[-1]):
            i -= 1
        left = runs_stack.pop(i - 2)
        right = runs_stack.pop(i - 2)
        runs_stack.insert(i - 2, merge(left, right))
    return runs_stack[0]


def intro_sort(array, depth=None):
    if depth is None:
        depth = int(math.log2(len(array)) * 2)
    if len(array) <= INTRO_SORT_INSERTION_SIZE:
        return insertion_sort(array)
    if depth == 0:
        return heap_sort(array)
    else:
        left = array[0]
        mid = array[len(array) // 2]
        right = array[-1]
        if left >= right:
            if right >= mid:
                sup = right
            elif left >= mid:
                sup = mid
            else:
                sup = left
        else:
            if right <= mid:
                sup = right
            elif left <= mid:
                sup = mid
            else:
                sup = left
        left_array = []
        right_array = []
        mid_array = []
        for el in array:
            if el < sup:
                left_array.append(el)
            elif el > sup:
                right_array.append(el)
            else:
                mid_array.append(el)
        return intro_sort(left_array, depth - 1) + mid_array + intro_sort(right_array, depth - 1)


def generate(N, min_el, max_el, mode):  # Генерация случайного списка
    new_array = [random.randint(min_el, max_el) for _ in range(N)]
    if mode == Modes.SORTED:
        new_array.sort(reverse=False)
    elif mode == Modes.BACK_SORTED:
        new_array.sort(reverse=True)
    elif mode == Modes.ALMOST_SORTED:
        new_array.sort(reverse=False)
        for i in range(N // 10):
            new_array[random.randint(0, N - 1)] = random.randint(min_el, max_el)
    return new_array


def check_sort(func):  # Получение времени работы сортировки и её правильности
    times = {}
    for mode in Modes:
        times[mode.value] = {}
        for N in POINTS:
            times[mode.value][N] = 0
            for i in range(REPEAT):
                array = generate(N, MIN_EL, MAX_EL, mode)
                ts = time.time()
                sorted_array = func(array)
                if times[mode.value][N] != -1:
                    times[mode.value][N] += time.time() - ts
                if sorted_array != sorted(array):
                    times[mode.value][N] = -1
            times[mode.value][N] /= REPEAT
            print(list(sorts.keys())[list(sorts.values()).index(func)], mode.value, N, times[mode.value][N])
    return times


class Modes(Enum):
    RANDOM = 'random'
    SORTED = 'sorted'
    BACK_SORTED = 'back sorted'
    ALMOST_SORTED = 'almost sorted'


sorts = {
    'selection': selection_sort,
    'insertion': insertion_sort,
    'bubble': bubble_sort,
    'merge': merge_sort,
    'quick': quick_sort,
    'shell': shell_sort,
    'shell_knuth': lambda array: shell_sort(array, 1),
    'shell_hibbard': lambda array: shell_sort(array, 2),
    'heap': heap_sort,
    'tim': tim_sort,  # ~ 40 times slower than built-in
    'intro': intro_sort,  # ~ 30 times slower than built-in
    'built-in': sorted,
    'not-working': lambda array: random.shuffle(array)  # to show all sorts are properly working
}
TIM_SORT_MIN_RUN = 64
TIM_SORT_GALOP = 7
INTRO_SORT_INSERTION_SIZE = 16
MIN_EL = 0  # Минимальное допустимое значение элемента
MAX_EL = 10**6  # Максимальное допустимое значение элемента
POINTS = [50] + list(range(10**3, 10**4 + 1, 10**3))  # Точки (количества элементов в списке для разных вызовов функции)
REPEAT = 10  # Times one point will be executed, then average from all results

if __name__ == "__main__":
    output = {"settings": {"REPEAT": REPEAT, "MIN_EL": MIN_EL, "MAX_EL": MAX_EL, "POINTS": POINTS}, "result": {}}
    print('SORT_METHOD\tARRAY_MODE\tN\tTIME, seconds')
    for key, sort in sorts.items():
        output["result"][key] = check_sort(sort)
    with open('result.json', 'w') as file:
        json.dump(output, file, indent=4)
    print('Successfully finished, check result.json')
