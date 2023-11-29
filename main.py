def main():
    while True:

        task = input("Enter the task number (0 to 5): ")

        if task == "0":
            import task0
            task0.main()
        elif task == "1":
            import task1
            task1.main()
        elif task == "2":
            import task2
            task2.run_task2()
        elif task == "3":
            import task3
            task3.main()
        elif task == "4":
            import task4
            task4.main()
        elif task == "5":
            import task4
            task4.main()
        elif task == "q":
            break
        else:
            print("Invalid task number. Please enter a number between 0 and 5.")
        print("\n")


if __name__ == "__main__":
    main()
