def scatter_plot_data(df, ax, title):
    c = df.iloc[:,2].map({0:'b',1:'black'})
    class_markers = df.iloc[:,2].map({0:'o',1:'<'})
    i = 0
    j = 0
    for _f1,_f2,c,cm,lb in zip(df.iloc[:,0],df.iloc[:,1],c,
                               class_markers,df.iloc[:,2]):
        if lb == 1:
            ax.scatter(_f1,_f2,marker = cm,c = c,s = 60,label = 'Class_1' if i == 0 else '')
            i +=1
        elif lb == 2:
            ax.scatter(_f1,_f2,marker = cm,c = c,s = 60,label = 'Class_2' if j == 0 else '')
            j +=1
    ax.set_title(title,size = 30)
    ax.set_xlabel('Feature 1',size = 20)
    ax.set_ylabel('Feature 2',size = 20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.legend(loc=0,prop={'size':15})

temp_df = pd.DataFrame(data = list((X_ros[:,0],X_ros[:,1],y_ros))).T
temp_df.columns = ['V1','V2','Class']
temp_df.head()

f, ax = plt.subplots(1, 1, sharey=False, figsize=(24,8))

scatter_plot_data(temp_df,ax,"Scatter Plot")
