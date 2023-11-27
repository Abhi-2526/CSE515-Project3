def main():
    task = input("Enter the task number (0 to 5): ")

    if task == "0":
        import task0
        task0.main()
    elif task == "1":
        k = int(input("Enter the number of latent semantics (k): "))
        import task1
        task1.run_task1(k)
    elif task == "2":
        import task2
        task2.run_task2()
    elif task == "3":
        # Implement task 3
        pass
    elif task == "4":
        import task4
        task4.main()
    elif task == "5":
        import task4
        task4.main()
    else:
        print("Invalid task number. Please enter a number between 0 and 5.")


if __name__ == "__main__":
    main()