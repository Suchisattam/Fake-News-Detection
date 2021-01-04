import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def trainer(model, train_dataloader, val_dataloader, num_epochs, path_to_save='/home/atharva',
		  checkpoint_path='/home/atharva',
		  checkpoint=100, train_batch=1, test_batch=1, device='cuda:0'): # 2 Marks. 
	"""
	Everything by default gets shifted to the GPU. Select the device according to your system configuration
	If you do no have a GPU, change the device parameter to "device='cpu'"
	:param model: the Classification model..
	:param train_dataloader: train_dataloader
	:param val_dataloader: val_Dataloader
	:param num_epochs: num_epochs
	:param path_to_save: path to save model
	:param checkpoint_path: checkpointing path
	:param checkpoint: when to checkpoint
	:param train_batch: 1
	:param test_batch: 1
	:param device: Defaults on GPU, pass 'cpu' as parameter to run on CPU. 
	:return: None
	"""
	torch.backends.cudnn.benchmark = True #Comment this if you are not using a GPU...
	model.train() # set the network to training mode.
	model.cuda()  # if gpu available otherwise comment this line. 
	# your code goes here.
	optimizer= torch.optim.Adam(model.parameters(), lr=1e-3)
	criterion= nn.CrossEntropyLoss().cuda()
	max_acc= None
	training_loss= []
	training_acc= []
	val_loss= []
	val_acc= []
	for epoch in range(num_epochs):
		#print("lol1")
		if (epoch%checkpoint)==0 and epoch!=0:
			torch.save({
				'epoch':epoch,
				'optimizer': optimizer.state_dict(),
				'model': model.state_dict(),
				'train_loss': training_loss,
				'val_loss': val_loss,
				'train_acc' : training_acc,
				'val_acc' : val_acc
			}, checkpoint_path+'/checkpoint.pt')
			torch.save(model.state_dict(), path_to_save+'/model_'+str(epoch)+'.pth')
			print(training_acc,training_loss,val_loss,val_acc)

		model.train()
		epoch_loss_train= 0
		epoch_acc_train=0
		for _, data in enumerate(train_dataloader):
			#print("lol2")
			optimizer.zero_grad()
			data["statement"]= data["statement"].cuda()
			data["justification"]= data["justification"].cuda()
			data["credit_history"]= data["credit_history"].cuda()
			data["label"]= data["label"].cuda()
			output= model(data["statement"],data["justification"],data["credit_history"])
			loss= criterion(output, data["label"])
			loss.backward()  
			optimizer.step()
			epoch_loss_train+= loss.item()
			__, predicted= torch.max(output.data, 1)
			epoch_acc_train+= (predicted == data["label"]).sum().item()
			del data
		training_loss.append(epoch_loss_train / _*train_batch) 
		training_acc.append(epoch_acc_train/_*train_batch)

		with torch.no_grad():  
			model.eval()  
			epoch_loss_val = 0
			epoch_acc_val = 0
			for _, data in enumerate(val_dataloader):
				#print("lol3")
				data["statement"]= data["statement"].cuda()
				data["justification"]= data["justification"].cuda()
				data["credit_history"]= data["credit_history"].cuda()
				data["label"]= data["label"].cuda()
				output= model(data["statement"],data["justification"],data["credit_history"])
				loss= criterion(output, data["label"])
				epoch_loss_val+= loss.item()
				__, predicted= torch.max(output.data, 1)
				epoch_acc_val+= (predicted == data["label"]).sum().item()

			val_loss.append(epoch_loss_val/_*test_batch)
			val_acc.append(epoch_acc_val/_*test_batch)

			if max_acc is None:
				max_acc= epoch_acc_val/_*test_batch
			else:
				if (epoch_acc_val/_*test_batch) > max_acc:
					print('saving at validation acc= ', epoch_acc_val/_*test_batch)
					torch.save(model.state_dict(), path_to_save+'/model.pth')



	
	
	plt.plot(training_acc)
	plt.plot(val_acc)
	plt.plot(training_loss)
	plt.plot(val_loss)
	plt.show()
	return None

