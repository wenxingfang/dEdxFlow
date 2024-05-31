"""Basic definitions for the flows module."""


import torch.nn

from nflows.distributions.base import Distribution
from nflows.utils import torchutils


class Flow(Distribution):
    """Base class for all flow objects."""

    def __init__(self, transform, distribution, embedding_net=None):
        """Constructor.

        Args:
            transform: A `Transform` object, it transforms data into noise.
            distribution: A `Distribution` object, the base distribution of the flow that
                generates the noise.
            embedding_net: A `nn.Module` which has trainable parameters to encode the
                context (condition). It is trained jointly with the flow.
        """
        super().__init__()
        self._transform = transform
        self._distribution = distribution
        if embedding_net is not None:
            assert isinstance(embedding_net, torch.nn.Module), (
                "embedding_net is not a nn.Module. "
                "If you want to use hard-coded summary features, "
                "please simply pass the encoded features and pass "
                "embedding_net=None"
            )
            self._embedding_net = embedding_net
        else:
            self._embedding_net = torch.nn.Identity()

    def _log_prob(self, inputs, context):
        embedded_context = self._embedding_net(context)
        noise, logabsdet = self._transform(inputs, context=embedded_context)
        log_prob = self._distribution.log_prob(noise, context=embedded_context)
        return log_prob + logabsdet

    def _sample(self, num_samples, context):
        #print("create an instance of:", self.__class__.__name__)
        #print('Flow:Hi _sample,',num_samples,context)
        embedded_context = self._embedding_net(context)
        noise = self._distribution.sample(num_samples, context=embedded_context)
        #print('Flow:Hi _sample, embedded_context=',embedded_context,",noise=",noise)

        if embedded_context is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            noise = torchutils.merge_leading_dims(noise, num_dims=2)
            embedded_context = torchutils.repeat_rows(
                embedded_context, num_reps=num_samples
            )

        #print('Flow:Hi noise=',noise,',embedded_context=',embedded_context)
        samples, _ = self._transform.inverse(noise, context=embedded_context)
        #print('Flow:Hi _sample, samples=',samples)

        if embedded_context is not None:
            # Split the context dimension from sample dimension.
            samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])
            #print('Flow:final samples, samples=',samples)

        return samples


    def jit_sample(self, noise, context):
        #print("create an instance of:", self.__class__.__name__)
        #print('Flow:Hi jit_sample,',num_samples,context)
        #embedded_context = self._embedding_net(context)
        embedded_context = context
        #noise = self._distribution.sample(num_samples, context=embedded_context)
        #print('Flow:Hi jit_sample, embedded_context=',embedded_context,",noise=",noise)

        #if embedded_context is not None:
        #    # Merge the context dimension with sample dimension in order to apply the transform.
        #    noise = torchutils.merge_leading_dims(noise, num_dims=2)
        #    embedded_context = torchutils.repeat_rows(
        #        embedded_context, num_reps=num_samples
        #    )

        #print('Flow:Hi noise=',noise,',embedded_context=',embedded_context)##noise (1,1), embedded_context (1,3)
        samples, _ = self._transform.inverse(noise, context=embedded_context)
        #print('Flow:Hi _sample, samples=',samples)

        #if embedded_context is not None:
        #    # Split the context dimension from sample dimension.
        #    samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])
        #    print('Flow:final samples, samples=',samples)

        return samples


    def forward(self, input_data):
        #input_data = input_data.double()
        #print("create an instance of:", self.__class__.__name__)
        embedded_context = input_data[:,0:3]
        noise            = input_data[:,3:4]
        #print('Flow:forward, embedded_context=',embedded_context,",noise=",noise)
        samples, _ = self._transform.inverse(noise, context=embedded_context)
        #print('Flow:Hi _sample, samples=',samples)
        samples = torchutils.inverse_logit(samples)
        #print('Flow:forward, samples=',samples)
        return samples

    def sample_and_log_prob(self, num_samples, context=None):
        """Generates samples from the flow, together with their log probabilities.

        For flows, this is more efficient that calling `sample` and `log_prob` separately.
        """
        embedded_context = self._embedding_net(context)
        noise, log_prob = self._distribution.sample_and_log_prob(
            num_samples, context=embedded_context
        )

        if embedded_context is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            noise = torchutils.merge_leading_dims(noise, num_dims=2)
            embedded_context = torchutils.repeat_rows(
                embedded_context, num_reps=num_samples
            )

        samples, logabsdet = self._transform.inverse(noise, context=embedded_context)

        if embedded_context is not None:
            # Split the context dimension from sample dimension.
            samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])
            logabsdet = torchutils.split_leading_dim(logabsdet, shape=[-1, num_samples])

        return samples, log_prob - logabsdet

    def transform_to_noise(self, inputs, context=None):
        """Transforms given data into noise. Useful for goodness-of-fit checking.

        Args:
            inputs: A `Tensor` of shape [batch_size, ...], the data to be transformed.
            context: A `Tensor` of shape [batch_size, ...] or None, optional context associated
                with the data.

        Returns:
            A `Tensor` of shape [batch_size, ...], the noise.
        """
        noise, _ = self._transform(inputs, context=self._embedding_net(context))
        return noise


class Flow2(Flow):##save calo flow to jit pt
    """Base class for all flow objects."""

    def __init__(self, transform, distribution, embedding_net=None):
        """Constructor.

        Args:
            transform: A `Transform` object, it transforms data into noise.
            distribution: A `Distribution` object, the base distribution of the flow that
                generates the noise.
            embedding_net: A `nn.Module` which has trainable parameters to encode the
                context (condition). It is trained jointly with the flow.
        """
        super().__init__(transform, distribution, embedding_net)

    def forward(self, input_data):
        #input_data = input_data.double()
        #print("create an instance of:", self.__class__.__name__)
        print('Flow2:input_data=',input_data)
        embedded_context = input_data[:,0:3]###mom, theta, recE
        noise            = input_data[:,3:28]## 5x5 noise
        #print('Flow:forward, embedded_context=',embedded_context,",noise=",noise)
        sample, _ = self._transform.inverse(noise, context=embedded_context)
        #print('Flow:Hi _sample, samples=',samples)
        ALPHA = 1e-6##(should match the ALPHA in data.py)
        sample = ((torch.sigmoid(sample) - ALPHA) / (1. - 2.*ALPHA))
        scaling = embedded_context[:,2:3]##recE
        sample = (sample / sample.abs().sum(dim=(-1), keepdims=True)) * scaling[:, 0].reshape(-1, 1)
        print('Flow2:sample=',sample)
        return sample

class FlowXT(Flow):##save x-t flow to jit pt
    """Base class for all flow objects."""

    def __init__(self, transform, distribution, embedding_net=None):
        """Constructor.

        Args:
            transform: A `Transform` object, it transforms data into noise.
            distribution: A `Distribution` object, the base distribution of the flow that
                generates the noise.
            embedding_net: A `nn.Module` which has trainable parameters to encode the
                context (condition). It is trained jointly with the flow.
        """
        super().__init__(transform, distribution, embedding_net)

    def forward(self, input_data):
        #input_data = input_data.double()
        #print("create an instance of:", self.__class__.__name__)
        embedded_context = input_data[:,0:2]
        noise            = input_data[:,2:3]
        #print('Flow:forward, embedded_context=',embedded_context,",noise=",noise)
        samples, _ = self._transform.inverse(noise, context=embedded_context)
        #print('Flow:Hi _sample, samples=',samples)
        samples = torchutils.inverse_logit(samples)
        #print('Flow:forward, samples=',samples)
        return samples
